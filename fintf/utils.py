import pandas_datareader.data as pdd
import datetime as dt
from pandas import HDFStore
import time
from . import settings
import stockstats as ss
from pandas_datareader._utils import RemoteDataError
import numpy as np
from uin_fc_lib import utils as uinutils


def put_to_storage(df=None, name=''):
    hdf = HDFStore(settings.storage_path)
    hdf.put(name, df, format='table', data_columns=True)


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
    hdf = HDFStore(settings.storage_path)
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
    df = None
    count = 0
    while df is None and count < 10:
        try:
            df = get_yahoo_data(start=start, end=end, symbol=yahoo_symbol)
        except RemoteDataError:
            time.sleep(10 + int(np.random.rand() * 10))
        count += 1

    if df is None:
        raise Exception('Even after 10 trials data could not be loaded from yahoo')

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


def get_symbol(symbol):
    symbol = clean_symbol(symbol)
    hdf = HDFStore(settings.storage_path)
    if symbol in hdf:
        return hdf[symbol]
    else:
        print('data from %s not in storage. You might want to load it with e.g. '
              'utils.get_past_10y_of_data(symbol)' % symbol)


def add_ti_and_store(symbol):
    symbol = clean_symbol(symbol)
    df = get_symbol(symbol)
    dfi = add_technical_indicators(df)
    put_to_storage(dfi, 't_%s' % symbol)
    return dfi


def get_tsymbol(symbol):
    symbol = clean_symbol(symbol)
    hdf = HDFStore(settings.storage_path)
    tsym = 't_%s' % symbol
    lprint('loaded %s from ti storage' % symbol)
    if tsym in hdf:
        return hdf[tsym]
    else:
        return add_ti_and_store(symbol)


import pickle
import os


def see_if_in_cache(key):
    fn = os.path.join(settings.data_path, key + '.pkl')
    if os.path.isfile(fn):
        return pickle.load(open(fn, 'rb'))
    # hdf = HDFStore(settings.proccess_cache)
    # if key in hdf:
    #    return hdf[key]


def put_in_cache(df, key):
    fn = os.path.join(settings.data_path, key + '.pkl')
    pickle._dump(df, open(fn, 'wb'))
    # hdf = HDFStore(settings.proccess_cache)
    # hdf.put(key, df, format='table', data_columns=True)


def assert_date_monotonic_increasing(df, date_column):
    if date_column in df:
        assert df[date_column].is_monotonic_increasing
    elif date_column == df.index.name:
        assert df.index.is_monotonic_increasing
    else:
        raise AttributeError('Date column not found on df')


def load_to_storage_from_file(filepath=None, symbol=None, df=None):
    if df is None:
        df = pd.read_csv(filepath)
    symbol = clean_symbol(symbol)
    df.columns = [x.replace(' ', '_') for x in df.columns]
    put_to_storage(
        df=df,
        name=symbol
    )


def add_technical_indicators(
        df,
        date_column='Date',
        col_names_for_olhcv=None,
        add_to_existing=False,
        indicators=None
):
    """
    add technical indicators to OLHC data.

    Args:
        df: pd.DataFrame
        col_names_for_olhcv: The column names needed for OLHC datat
        add_to_existing: is data is joined to an existting dataframe
        indicators:

    """

    if date_column in df:
        assert df[date_column].is_unique
        assert df[date_column].is_monotonic_increasing
    elif date_column == df.index.name:
        assert df.index.is_unique
        assert df.index.is_monotonic_increasing
    else:
        raise AttributeError('Date column not found')

    indicators = indicators if indicators is not None else \
        ['atr', 'tr', 'cci_20', 'rsv_30', 'rsv_60', 'rsv_12', 'rsv_7', 'rsv_5', 'wr_12',
         'macd', 'rsi_14', 'wr_3', 'wr_5', 'wr_7', 'wr_10', 'wr_14', 'rsi_5', 'rsi_60',
         'rsi_30', 'rsi_3', 'dma', 'cci', 'kdjd', 'pdi', 'dx']
    h_data = df.copy()

    # rename columns such that they match the olhcv paradigm from yahoo
    if col_names_for_olhcv:
        rename_dict = {}
        for key in col_names_for_olhcv:
            rename_dict[col_names_for_olhcv[key]] = key
        h_data.rename(columns=rename_dict, inplace=True)

    stock = ss.StockDataFrame.retype(h_data)
    for ti in indicators:
        stock.get(ti)

    indicators.append('close')

    h_data = h_data[indicators].copy()

    # add momentum variables
    for p in [5, 10, 50, 60, 100, 200]:
        h_data['mom_%d' % p] = h_data['close'].diff(p)

    # add moving averages
    for ma in [5, 10, 20, 50, 100, 200]:
        for col in ['mom_60', 'mom_10']:
            h_data['%s_ma_%d' % (col, ma)] = h_data[col].rolling(ma).mean()

    for p in [1, 10, 20]:
        h_data['ret_%dd' % p] = h_data['close'].pct_change(p).shift(-p)
        indicators.append('ret_%dd' % p)

    return h_data




def join_unemployment_data(df):
    # Join additional data sources, e.g. unemployment, gold as an indicator for demand of security, Short-Term and Long Term Bond Yields
    unemployment = pd.read_csv('C:/Users/AL_BC/Downloads/DP_LIVE_18012018171856485.csv')
    # select monthly US unemployment as relative to workforce figure
    unemployment = unemployment[
        (unemployment.LOCATION == 'USA') & (unemployment.SUBJECT == 'TOT') & (unemployment.MEASURE == 'PC_LF') & (
                unemployment.FREQUENCY == 'M')]
    unemployment.TIME = pd.to_datetime(unemployment.TIME)
    unemployment.set_index('TIME', inplace=True)
    dft = pd.concat([df, unemployment['Value']], axis=1)
    dft.Value.fillna(method='ffill', inplace=True)
    dft.dropna(inplace=True)
    dft.rename(columns={'Value': 'unemp_rate'}, inplace=True)
    return dft
