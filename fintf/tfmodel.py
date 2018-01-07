from uin_fc_lib import ts_forecasts, ml_visualizations
import pandas as pd
import numpy as np
import keras as k
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from . import utils
import hashlib

class TF_LSTM_Regressor(object):
    def __init__(self, input_dim, validation_ratio=.3, look_back=1):
        # fix random seed for reproducibility
        self.look_back = look_back
        self.validation_ratio = validation_ratio
        seed = 7
        np.random.seed(seed)
        # evaluate model with standardized dataset
        self.input_dim = input_dim
        print(self.input_dim)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4,
                                           mode='min')
        mcp_save = ModelCheckpoint('md.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        estimators = []
        #estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(
            build_fn=self.baseline_model,
            epochs=50,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping, reduce_lr_loss],  # , mcp_save],
            validation_split=self.validation_ratio
        )))
        self.pipeline = Pipeline(estimators)
        print('model compiled')

    # convert an array of values into a dataset matrix

    def baseline_model(self):
        # create and fit the LSTM network
        model = k.models.Sequential()
        model.add(k.layers.LSTM(4, input_shape=self.input_dim))
        model.add(k.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def predict(self, X):
        return self.pipeline.predict(X)

    def fit(self, X, y):
        self.pipeline.fit(X, y)


class TF_Regressor1(object):

    def __init__(self, input_dim, validation_ratio=.3):
        # fix random seed for reproducibility
        self.validation_ratio = validation_ratio
        seed = 7
        np.random.seed(seed)
        # evaluate model with standardized dataset
        self.input_dim = input_dim
        print(self.input_dim)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4,
                                           mode='min')
        mcp_save = ModelCheckpoint('md.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(
            build_fn=self.baseline_model,
            epochs=50,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping, reduce_lr_loss],  # , mcp_save],
            validation_split=self.validation_ratio
        )))
        self.pipeline = Pipeline(estimators)

    def baseline_model(self):
        model = k.models.Sequential()
        model.add(k.layers.Dense(32, kernel_initializer='normal', input_dim=self.input_dim))
        model.add(k.layers.Dropout(.2))
        model.add(k.layers.Activation('relu'))
        model.add(k.layers.Dense(1, kernel_initializer='normal'))
        # also possible is mean_squared_error
        #

        model.compile(
            optimizer='adam',
            loss='mean_absolute_error',
        )
        return model

    def predict(self, X):
        return self.pipeline.predict(X)

    def fit(self, X, y):
        self.pipeline.fit(X, y)


def headline_of_X(df, hide_columns, date_column, target):
    drop_cols = hide_columns
    drop_cols.append(date_column)
    drop_cols.append(target)
    drop_cols.append('index')
    unnamed_cols = df.columns[df.columns.str.startswith('Unnamed:')]
    drop_cols.extend(unnamed_cols)
    return df.columns[~df.columns.isin(drop_cols)]


def train_tf_regressor1_model(
        df=None,
        date_column=None,
        backtest_settings=None,
        target=None,
        hide_columns=None,
        validation_ratio=0
):
    if backtest_settings is None:
        backtest_settings = {}
    input_dim = len(headline_of_X(df=df, target=target, date_column=date_column, hide_columns=hide_columns))
    # subtract target
    model = TF_Regressor1(input_dim=input_dim, validation_ratio=validation_ratio)
    tfc = ts_forecasts.TFC(df=df, date_column=date_column)
    tfc.train_model(target=target, hide_columns=hide_columns, model=model, **backtest_settings)
    return tfc



def train_lstm_regressor_model(
        df=None,
        date_column=None,
        backtest_settings=None,
        target=None,
        hide_columns=None,
        validation_ratio=0,
        look_back=5
):
    df.dropna(inplace=True)
    if backtest_settings is None:
        backtest_settings = {}
    if hide_columns is None:
        hide_columns = []
    input_dim = len(headline_of_X(df=df, target=target, date_column=date_column,
                                  hide_columns=hide_columns))

    print(input_dim)

    model = TF_LSTM_Regressor(input_dim=(input_dim, look_back),
                              validation_ratio=validation_ratio)
    lstm_columns = hide_columns

    lstm_columns.append(target)
    lstm_columns.pop(lstm_columns.index(date_column))
    print(lstm_columns)
    X = create_LSTM_dataset(df[df.columns[~df.columns.isin(hide_columns)]],
                            look_back=look_back,
                            date_column=date_column)

    tfc = None
    # tfc = ts_forecasts.TFC(df=X, date_column=date_column)
    # tfc.train_model(target=target, hide_columns=hide_columns, model=model, **backtest_settings)
    y = df[target][look_back : ]
    model.fit(X, y)
    return X,  y#atfc

def get_key_for_lstm_dataset(df, date_column, look_back):
    s = df.columns.__str__() + \
        df[df.columns[0]].__str__() + \
        df[df.columns[-1]].__str__()
    s = s.encode('utf-8')
    s = hashlib.sha224(s).hexdigest()
    return '%s_%s_%s' % (s, date_column, look_back)


def create_LSTM_dataset(df, date_column, look_back=1):

    if not df.index.name == date_column:
        df.set_index(date_column, inplace=True)
    rm = df.ewm(halflife=100).mean()
    rstd = df.ewm(halflife=100).std()
    df = (df - rm) / rstd

    df.dropna(inplace=True)
    key = get_key_for_lstm_dataset(df, date_column, look_back)
    df_store = None
    df_store = utils.see_if_in_cache(key)
    if df_store is not None:
        print('processed data loaded from cache')
        return df_store
    utils.assert_date_monotonic_increasing(df=df, date_column=date_column)
    if date_column == df.index.name:
        df.reset_index(inplace=True)
    dataX = []
    for i in range(df.shape[0] - look_back + 1):
        a = df.values[i:(i + look_back), :]
        dataX.append(a)
    X = np.array(dataX)
    X = np.reshape(X, (X.shape[0], look_back, X.shape[2]))

    q = pd.DataFrame()
    q[date_column] = df[date_column][:df.shape[0] - look_back + 1]
    q.index.name = '__enum__'
    q.reset_index(inplace=True)
    for i, col in enumerate(df.columns):
        if col == date_column:
            continue
        q[col] = q.__enum__.map(lambda num: X[num, :, i])
    q.drop('__enum__', inplace=True, axis=1)
    q.set_index(date_column, inplace=True)

    n = q.values
    # reshape the Matrix such that it matches the numpy shape
    x = np.zeros((n.shape[0],  n.shape[1], look_back))

    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            x[i, j, :] = n[i, j]
            x[i, j, :] = n[i, j]
    utils.put_in_cache(x, key)
    return x


def main():
    df = pd.read_csv('test_data3.csv')
    backtest_settings = {
        'backtest_method': 'walk_forward_rolling'

    }
    hide_columns = ['regression_target', 'Close', 'target', 'ret_1d']
    # tfc = train_tf_regressor1_model(df=df, date_column = 'Date', target='regression_target', backtest_settings=backtest_settings, hide_columns=hide_columns)
    tfc = train_lstm_regressor_model(df=df, date_column='Date', target='regression_target',
                                     backtest_settings=backtest_settings, hide_columns=hide_columns)
    return tfc


# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

if __name__ == '__main__':
    tfc = main()
    # ml_visualizations.run_graphics(tfc)
