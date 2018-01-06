from .. import utils
import os
from .. import settings

def refresh_data_for_symbols():
    symbols = ['SPY', 'VXX', 'GDX', 'XIV', 'EURUSD=x']
    for symbol in symbols:
        print('loading %s' % symbol)
        utils.get_past_5y_of_data(symbol)
        utils.add_ti_and_store(symbol)

def initialize_data_for_symbols():
    os.remove(settings.storage_path)
    symbols = ['SPY', 'VXX', 'GDX', 'XIV', 'EURUSD=x']
    for symbol in symbols:
        print('loading %s' % symbol)
        utils.get_past_10y_of_data(symbol)
        utils.add_ti_and_store(symbol)