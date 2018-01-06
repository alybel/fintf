from .. import utils

def load_data_for_symbols():
    symbols = ['SPY', 'VXX', 'GDX', 'XIV', 'EURUSD=x']
    for symbol in symbols:
        utils.get_past_5y_of_data(symbol)