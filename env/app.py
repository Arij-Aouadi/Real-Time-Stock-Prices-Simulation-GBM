from threading import Lock, Event
from flask import Flask, render_template, session,request
from flask_socketio import SocketIO, emit
import requests
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import json

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()
display_prices_event = Event()  # Event to control whether to display prices or not


url = 'https://api.coinbase.com/v2/prices/btc-usd/spot'
end_date=''
start_date=''
symbol=[]
numberOfSimulatedDays=0
numberOfSimulatedPricesPerSec=0
count=0

def calculate_mu_sigma(symbole, start_date, end_date):
    # Get the last month's historical data
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    # Fetch historical data using yfinance
    stock_data = yf.download(symbole, start=start_date, end=end_date)

    # Calculate daily returns
    returns = (stock_data['Close'].pct_change()).dropna()

    # Calculate mu and sigma
    mu = np.mean(returns)
    sigma = np.std(returns)

    return mu, sigma

def simulate_stock_price_with_orders(n, T, M, mu, sigma, initial_price, Sf, Sh, Sl,count, order_size=0, order_direction=1):
    dt = T/n
    rand = np.random.normal(0, np.sqrt(dt), size=(M,n))

    # Adjust drift based on order impact
    mu += order_direction * order_size / (initial_price * M * n)


    St = np.exp(
        (mu - sigma ** 2 / 2 ) * dt
        + sigma * np.sqrt(dt) * rand 
    ).T

    St = np.vstack([np.ones(M), St])
    St = initial_price * St.cumprod(axis=0)
    cliped = np.clip(St[1], Sl, Sh)
    modified_array = np.concatenate(([initial_price], cliped))

    return modified_array[count]


def background_thread():
    """Example of how to send server generated events to clients."""
    global count
    while True:
        display_prices_event.wait()  # Wait for the event to be set
        mu,sigma=calculate_mu_sigma("GOOGL","2022-10-5","2022-10-9")
        simulated_price=simulate_stock_price_with_orders(1, 1, 598, mu, sigma, 120.9, 126.05, 127.54, 114.37, count,order_size=0, order_direction=1)
        socketio.sleep(3)
        count += 1
        price = ((requests.get(url)).json())['data']['amount']
        socketio.emit('my_response',
                      {'data': 'Bitcoin current price (USD): ' + str(simulated_price), 'count': count})

@app.route('/')
def index():
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
