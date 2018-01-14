import pandas_datareader.data as pdd
import datetime as dt
from pandas import HDFStore
import time
from . import settings
import stockstats as ss
from pandas_datareader._utils import RemoteDataError
import numpy as np


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


from uin_fc_lib import ts_forecasts
import pandas as pd


class MetaModel(object):
    def __init__(self,
                 df=None,
                 target_dict=None,
                 hide_columns=None,
                 backtest_settings=None,
                 date_column=None,
                 model_cascade=None,
                 specific_model_settings=None
                 ):
        self.df = df.copy()
        if not isinstance(target_dict, dict):
            raise AttributeError('Targets need to be provided as a dictionary')
        self.target_dict = target_dict

        # Check if each model has a target
        target_vals = []
        for key in target_dict:
            target_vals.extend(target_dict[key])
        for layer in model_cascade:
            for model_name in layer:
                if model_name not in target_vals:
                    raise AttributeError('No Target found for %s' % model_name)
        if pd.Series(target_vals).duplicated().any():
            raise AttributeError('Target Dictionary contains duplicates')

        # Check if each target is on the data
        for target in target_dict:
            if target not in self.df:
                raise AttributeError('Target %s not found on data Frame' % target)

        self.hide_columns = hide_columns
        self.backtest_settings = backtest_settings
        self.date_column = date_column
        self.model_cascade = model_cascade
        self.results = {}

        self.specific_model_settings = {}
        if specific_model_settings is not None:
            self.specific_model_settings = specific_model_settings

    @staticmethod
    def _run_in_loop(
            model,
            df,
            target,
            hide_columns,
            backtest_settings,
            date_column,
            specific_model_settings=None
    ):
        tfc = ts_forecasts.TFC(date_column=date_column, df=df)
        if specific_model_settings is not None:
            backtest_settings = specific_model_settings
        tfc.train_model(model=model, target=target, hide_columns=hide_columns, **backtest_settings)
        return tfc

    def run(self):
        data = None
        for i, layer in enumerate(self.model_cascade):
            print('running layer %d', i)
            this_layer_results = {}
            for model in layer:
                specific_model_settings = None
                if model in self.specific_model_settings and self.specific_model_settings[model] is not None:
                    specific_model_settings = self.specific_model_settings[model]
                target_col = None
                for key in self.target_dict:
                    if model in self.target_dict[key]:
                        target_col = key
                tfc = self._run_in_loop(
                    model=layer[model],
                    df=self.df,
                    hide_columns=self.hide_columns,
                    backtest_settings=self.backtest_settings,
                    target=target_col,
                    date_column=self.date_column,
                    specific_model_settings=specific_model_settings
                )
                this_layer_results[model] = tfc
                self.results[model] = tfc

            # do not use insample forecasts, this leads to very wrong results
            for trained_model in this_layer_results:
                data = this_layer_results[trained_model].all_data_with_predictions

                target_col = None
                for key in self.target_dict:
                    if trained_model in self.target_dict[key]:
                        target_col = key

                # This is the Prob Case
                if 'prob' in data:
                    data.rename(columns={'prob': '%s_%s_prob' % (trained_model, target_col)}, inplace=True)
                    print('joined %s' % ('%s_%s_prob' % (trained_model, target_col)))
                    self.df = self.df.join(data['%s_%s_prob' % (trained_model, target_col)])

                # This is the Regression Case
                else:
                    data.rename(columns={'pred': '%s_%s_pred' % (trained_model, target_col)}, inplace=True)
                    self.df = self.df.join(data['%s_%s_pred' % (trained_model, target_col)])

                # clean self.df
                self.df.dropna(inplace=True)
                self.df = self.df.join(data['__oos__'])
                self.df = self.df[self.df['__oos__'] == 1]
                self.df.drop(['__oos__'], axis=1, inplace=True)

    def get_results(self):
        return self.results

    def get_data(self):
        return self.df
