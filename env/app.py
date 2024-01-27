from threading import Lock, Event
from flask import Flask, render_template, session,request
from flask_socketio import SocketIO, emit
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
numberOfSimulatedOptionStrikesPerDate=0
numberOfsimulatedOptionDates=0
intervalleBetweenTwoDatesInDays=0
stepOfStrike=0
count=0
days=3
symbols_to_simulate = ["GOOGL", "AAPL", "MSFT", "AMZN", "Meta","PYPL","NVDA","TSLA","ORCL","SHEL"] 
global_sim_prices=[]
global_closing_prices=[]

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
    # Get the closing price
    closing_price = stock_data['Adj Close'].iloc[-1]
    return mu, sigma, initial_price, closing_price

# MonteCarlo simulation function for Geometric Brownian Motion or Jump Diffusion
def monte_carlo_simulation(num_simulations, time_steps, mu, sigma, initial_price, diffusion_type='GBM', mu_j=0, sigma_j=0, lambda_=0):
    global day 
    global global_sim_prices
    global global_variation
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
    global_sim_prices.append(np.mean(sim_prices,axis=0))

def black_scholes_Option_Pricing(S,K,T,r,q,sigma):
    """
    Inputs
    #S = Current stock Price
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #r = risk free interest rate
    #q = dividend yield
    # sigma = volatility 
    
    Output
    # call_price = value of the option 
    """
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    
    #call = S * np.exp(-q*T)* np.norm.cdf(d1) - K * np.exp(-r*T)*np.norm.cdf(d2)
    #put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


    return d1
    

def background_thread():
    """Example of how to send server generated events to clients."""
    global count
    global symbols_to_simulate
    global day
    while True:
        display_prices_event.wait()  # Wait for the event to be set
        simulated_prices = []
        variation=[]
        variationEnPorcentage=[]
        if (count==0):
            for i in range(len(symbols_to_simulate)):
                start_date = '2010-01-01'
                num_simulations = 1000
                time_steps = 600
                diffusion_type ='GBM' # or 'Jump' and if jump, add mu_j, sigma_j, lambda_ to the functionsocketio.sleep(3)
                mu, sigma, initial_price, closing_price = get_stock_data(symbols_to_simulate[i], start_date)
                global_closing_prices.append(closing_price)
                monte_carlo_simulation(num_simulations, time_steps, mu, sigma, closing_price, diffusion_type)
                simulated_prices.append(round(global_sim_prices[i][count],3))
                variation.append(round(global_sim_prices[i][count]-global_closing_prices[i],2))
                variationEnPorcentage.append(round(((global_sim_prices[i][count]-global_closing_prices[i])/global_closing_prices[i])*100,2))
        else :
            for i in range (len(symbols_to_simulate)):
                simulated_prices.append(round(global_sim_prices[i][count],3))
                variation.append(round(global_sim_prices[i][count]-global_closing_prices[i],2))
                variationEnPorcentage.append(round(((global_sim_prices[i][count]-global_closing_prices[i])/global_closing_prices[i])*100,2))
        if (count==600) :
            day=day+1
            count=0
               
        socketio.sleep(1) 
        count += 1
        socketio.emit('my_response',
                      {'data': simulated_prices, 'count': count,'variation' : variation,'variationEn': variationEnPorcentage})

@app.route('/')
def index():
    print(sys.path)
    return render_template('index.html', async_mode=socketio.async_mode)

@app.route('/update_variables', methods=['POST'])
def update_variables():
    global end_date, start_date, symbol, numberOfSimulatedDays, numberOfSimulatedPricesPerSec,numberOfSimulatedOptionStrikesPerDate,numberOfsimulatedOptionDates

    data = request.json

    # Update variables based on the JSON data received in the POST request
    end_date = data.get('end_date', '')
    start_date = data.get('start_date', '')
    symbol = data.get('symbol', [])
    numberOfSimulatedDays = data.get('numberOfSimulatedDays', 0)
    numberOfSimulatedPricesPerSec = data.get('numberOfSimulatedPricesPerSec', 0)
    numberOfSimulatedOptionStrikesPerDate = data.get('numberOfSimulatedOptionStrikesPerDate',0)
    numberOfsimulatedOptionDates = data.get('numberOfsimulatedOptionDates',0)
    
    return {'message': 'Variables updated successfully',}

@app.route('/get_variables', methods=['GET'])
def get_variables():
    global end_date, start_date, symbol, numberOfSimulatedDays, numberOfSimulatedPricesPerSec,numberOfSimulatedOptionStrikesPerDate,numberOfsimulatedOptionDates


    return {
    'end_date': end_date,
    'start_date': start_date,
    'symbol': symbol,
    'numberOfSimulatedDays': numberOfSimulatedDays,
    'numberOfSimulatedPricesPerSec': numberOfSimulatedPricesPerSec,
    'numberOfSimulatedOptionStrikesPerDate' : numberOfSimulatedOptionStrikesPerDate,
    'numberOfsimulatedOptionDates' : numberOfsimulatedOptionDates
}
@app.route('/getInitialPrices', methods=['GET'])
def get_Initial():
    global global_closing_prices
    return {
    'initial': global_closing_prices}
@app.route('/control_simulation', methods=['POST'])
def control_simulation():
    data = request.json
    order = data.get('order', '')
    global count
    global day
    if order == 'start':
        display_prices_event.set()
        # Reset count to 0 when variables are updated
        count = 0
        day = 1
        return {'message': 'Simulation started successfully'}
    elif order == 'freeze':
        display_prices_event.clear()
        return {'message' :'simulation frozen successfully'}
    elif order == 'continue':
        display_prices_event.set()
        return {'message' :'simulation continued successfully'}
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

@socketio.on('control_simulation')
def control_simulation(data):
    order = data.get('order', '')
    global count
    global day
    if order == 'start':
        display_prices_event.set()
        # Reset count to 0 when variables are updated
        count = 0
        day = 1
        socketio.emit('simulation_status', {'message': 'Simulation started successfully'})
    elif order == 'freeze':
        display_prices_event.clear()
        socketio.emit('simulation_status', {'message': 'Simulation frozen successfully'})
    elif order == 'continue':
        display_prices_event.set()
        socketio.emit('simulation_status', {'message': 'Simulation continued successfully'})
    elif order == 'stop':
        display_prices_event.clear()
        socketio.emit('simulation_status', {'message': 'Simulation stopped successfully'})
    else:
        socketio.emit('simulation_status', {'message': 'Invalid order. Use {"order": "start"} or {"order": "stop"}'})

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
    emit('my_response', {'data': '', 'count': 0,'variation' : '','variationEn': ''})

if __name__ == '__main__':
    socketio.run(app)
