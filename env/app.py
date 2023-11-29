from threading import Lock, Event
from flask import Flask, render_template, session,request
from flask_socketio import SocketIO, emit
import requests
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import json
import datetime as dt
from flask_cors import CORS
import sys 


# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")
thread = None
thread_lock = Lock()
display_prices_event = Event()  # Event to control whether to display prices or not


end_date=''
start_date=''
symbol=[]
numberOfSimulatedDays=0
numberOfSimulatedPricesPerSec=0
count=0
day=0
symbols_to_simulate = ["GOOGL", "AAPL", "MSFT", "AMZN", "Meta","PYPL","NVDA","TSLA","ORCL","SHEL"] 


# Retreive stock data from desired start date until last available
def get_stock_data(ticker, start_date):
    today = dt.date.today()
    end_date = today.strftime("%Y-%m-%d")
    try:
        # Get the stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
    except:
      return "No data found, please try again later"
    # Calculate the daily logarithmic returns
    returns = np.log(stock_data['Adj Close']/stock_data['Adj Close'].shift(1))
    returns = returns.dropna()
    # Calculate the mean rate of return (mu)
    mu = returns.mean()
    # Calculate the volatility (sigma)
    sigma = returns.std()
    # Get the initial price
    initial_price = stock_data['Adj Close'][-1]
    return mu, sigma, initial_price

# MonteCarlo simulation function for Geometric Brownian Motion or Jump Diffusion
def monte_carlo_simulation(num_simulations, time_steps, mu, sigma, initial_price, diffusion_type='GBM', mu_j=0, sigma_j=0, lambda_=0):
    global day    
    # Create an array to store the simulated prices
    sim_prices = np.zeros((num_simulations, time_steps))
    # Set the initial price for all simulations
    sim_prices[:, 0] = initial_price
    # Set the time step
    dt = 1 / time_steps
    # Choose the type of diffusion
    if diffusion_type == 'GBM':
        for i in range(1, time_steps):
            # Generate random numbers for each simulation
            rand = np.random.normal(0, 1, num_simulations)
            # Use GBM to calculate the next price for each simulation
            sim_prices[:, i] = sim_prices[:, i-1] * np.exp((mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * rand)
    elif diffusion_type == 'Jump':
        for i in range(1, time_steps):
            # Generate random numbers for each simulation
            rand = np.random.normal(0, 1, num_simulations)
            # Generate random numbers for the jump component
            jump = np.random.normal(mu_j, sigma_j, num_simulations)
            # Use Jump Diffusion to calculate the next price for each simulation
            sim_prices[:, i] = sim_prices[:, i-1] * np.exp((mu - sigma**2 / 2 - lambda_ * (np.exp(mu_j + sigma_j ** 2 / 2) - 1)) * dt + sigma * np.sqrt(dt) * rand + jump)
    else:
        raise ValueError('Invalid diffusion type. Choose either GBM or Jump.')
    return round(sim_prices[day][count],3)


def background_thread():
    """Example of how to send server generated events to clients."""
    global count
    while True:
        display_prices_event.wait()  # Wait for the event to be set

        simulated_prices = []
        for symbol in symbols_to_simulate:
            
            start_date = '2010-01-01'
            num_simulations = 5
            time_steps = 600
            diffusion_type ='GBM' # or 'Jump' and if jump, add mu_j, sigma_j, lambda_ to the functionsocketio.sleep(3)
            mu, sigma, initial_price = get_stock_data(symbol, start_date)
            simulated_price = monte_carlo_simulation(num_simulations, time_steps, mu, sigma, initial_price, diffusion_type)
            simulated_prices.append(simulated_price)


        socketio.sleep(3) 
        count += 1
        socketio.emit('my_response',
                      {'data': simulated_prices, 'count': count})

@app.route('/')
def index():
    print(sys.path)
    return render_template('index.html', async_mode=socketio.async_mode)

@app.route('/update_variables', methods=['POST'])
def update_variables():
    global end_date, start_date, symbol, numberOfSimulatedDays, numberOfSimulatedPricesPerSec

    data = request.json

    # Update variables based on the JSON data received in the POST request
    end_date = data.get('end_date', '')
    start_date = data.get('start_date', '')
    symbol = data.get('symbol', [])
    numberOfSimulatedDays = data.get('numberOfSimulatedDays', 0)
    numberOfSimulatedPricesPerSec = data.get('numberOfSimulatedPricesPerSec', 0)
    
    return {'message': 'Variables updated successfully',}

@app.route('/get_variables', methods=['GET'])
def get_variables():
    global end_date, start_date, symbol, numberOfSimulatedDays, numberOfSimulatedPricesPerSec

    return {
    'end_date': end_date,
    'start_date': start_date,
    'symbol': symbol,
    'numberOfSimulatedDays': numberOfSimulatedDays,
    'numberOfSimulatedPricesPerSec': numberOfSimulatedPricesPerSec
}
@app.route('/control_simulation', methods=['POST'])
def control_simulation():
    data = request.json
    order = data.get('order', '')
    global count

    if order == 'start':
        display_prices_event.set()
        # Reset count to 0 when variables are updated
        count = 0
        return {'message': 'Simulation started successfully'}
    elif order == 'stop':
        display_prices_event.clear()
        return {'message': 'Simulation stopped successfully'}
    else:
        return {'message': 'Invalid order. Use {"order": "start"} or {"order": "stop"}'}


@socketio.event
def my_event(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})

# Receive the test request from client and send back a test response
@socketio.on('test_message')
def handle_message(data):
    print('received message: ' + str(data))
    emit('test_response', {'data': 'Test response sent'})

# Broadcast a message to all clients
@socketio.on('broadcast_message')
def handle_broadcast(data):
    print('received: ' + str(data))
    emit('broadcast_response', {'data': 'Broadcast sent'}, broadcast=True)

@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})

if __name__ == '__main__':
    socketio.run(app)
