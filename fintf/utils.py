import pandas_datareader.data as pdd
import datetime as dt
from pandas import HDFStore


def lprint(x):
    print(x)


def get_yahoo_data(start=None, end=None, symbol=None):
    return pdd.DataReader(symbol, 'yahoo', start=start, end=end)


def get_yahoo_quote(symbol=None):
    return pdd.get_quote_google(symbol)

def clean_symbol(symbol):
    symbol = symbol.replace('=x', '')
    return symbol

def load_from_store_or_yahoo(start=None, end=None, symbol=None):
    append = False
    hdf = HDFStore('financial_data_storage.h5')
    today = dt.datetime.today().date()

    yahoo_symbol = symbol
    symbol = clean_symbol(symbol)

    # this case, earlier data than in store is requested. The table needs to be rewritten
    if symbol in hdf:
        df = hdf[symbol]
        start_store = df.index.min()
        if isinstance(start, str):
            start = dt.datetime.strptime(start, '%Y-%m-%d')
        if start_store.date() > start:
            hdf.remove(symbol)
            lprint('start date was earlier than the oldest date in the storage. storage needs to be rewritten.')

    if symbol in hdf:
        df = hdf[symbol]
        end_store = df.index.max()

        # check if today is a weekend day
        weekday = dt.datetime.today().weekday()
        last_trading_day = today
        if weekday in [5, 6]:
            correction = 1 if weekday == 5 else 2
            last_trading_day = today - dt.timedelta(correction)

        # if the last trading day is the max date in the store than do not reload data
        if last_trading_day == end_store.date():
            lprint('loaded %s data from storage.' % symbol)
            return df

        # if the last trading is younger that the last trading day, load the difference
        end = today + dt.timedelta(1)
        start = end_store
        append = True

    # if no store was found, use the start and end from above
    df = get_yahoo_data(start=start, end=end, symbol=yahoo_symbol)
    # remove blanks in the header
    df.columns = [x.replace(' ', '_') for x in df.columns]

    # store or append to hdf5 storage

    if append:
        hdf.append(symbol, df, format='table', data_columns=True)
    else:
        hdf.put(symbol, df, format='table', data_columns=True)
    if not df.index.is_unique:
        lprint('index of %s is not unique' % symbol)
    return df


# function to get the past 10y of daily data
def get_past_10y_of_data(symbol):
    today = dt.datetime.today().date()
    start = today - dt.timedelta(10 * 365)
    end = today + dt.timedelta(1)
    return load_from_store_or_yahoo(start, end, symbol)

def get_past_5y_of_data(symbol):
    today = dt.datetime.today().date()
    start = today - dt.timedelta(5 * 365)
    end = today + dt.timedelta(1)
    return load_from_store_or_yahoo(start, end, symbol)
