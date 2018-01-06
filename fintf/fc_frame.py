# author: Alexander Beck

import pandas as pd
from inspect import signature
import numpy as np
from collections import defaultdict

class FF(object):
    def __init__(self, df=None, date_column=None):
        self.backtest_method = None
        self.y = None
        self.df = df
        self.date_column = date_column
        self.execution_lag = 0
        self.training_results = defaultdict(list)
        self.in_sample_results = defaultdict(list)

    def _prepare_X_and_y_data(self, hide_columns):
        # For each experiment, take a fresh copy of the provided data
        self.df.dropna(inplace=True)
        self.df_h = self.df.reset_index().copy()
        self.Date = self.df_h[self.date_column].copy()
        self.df_h.set_index(self.date_column, inplace=True)
        self.unique_dates = self.df_h.index.unique()

        # In case when there is an execution lag, shift the target vector by the execution lag and remove all
        # nan columns that occur
        if self.execution_lag > 0:
            self.df_h[self.target] = self.df_h[self.target].shift(-self.execution_lag)
            self.df_h.dropna(inplace=True)

        self.y = self.df_h[self.target].copy()

        # change into into numeric series
        self.y = self.y.astype(np.float64)

        drop_cols = [self.target, self.date_column, 'index']
        drop_cols.extend(hide_columns)

        self.X = self.df_h.drop(drop_cols, axis=1, errors='ignore').copy()

        assert (self.X.shape[0] == self.df.shape[0] - self.execution_lag)


    def train(self,
              model=None,
              backtest_method='walk_forward_rolling',
              target=None,
              hide_columns=None,
              split_ratio=.7,
              training_window=1000,
              oos_window=10,
              step_size=10,
              test_train_diff_days=0,
              sample_weights=None,
              kwargs=None):
        if hide_columns is None:
            hide_columns = []
        self.training_window = training_window
        self.oos_window = oos_window
        self.test_train_diff_days = test_train_diff_days
        self.split_ratio = split_ratio
        self.step_size = step_size
        self.target=target
        self.sample_weight = sample_weights
        self.clf = model
        self._prepare_X_and_y_data(hide_columns=hide_columns)

        self.backtest_method = backtest_method



        if self.backtest_method == 'simple_split':
            cutoff = int(self.X.shape[0] * self.simple_split_ratio)

            X_train, X_test = self.X.loc[self.unique_dates[:cutoff]], self.X.loc[self.unique_dates[cutoff:]]
            y_train, y_test = self.y.loc[self.unique_dates[:cutoff]], self.y.loc[self.unique_dates[cutoff:]]
            sample_weight_train, sample_weights_test = None, None
            if self.sample_weight is not None:
                sample_weight_train = sample_weights[:cutoff]
                sample_weight_test = sample_weights[cutoff:]

            self._train_clf(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                            run_no=i, sample_weight=sample_weight_train)

        elif self.backtest_method in ['walk_forward_extending', 'walk_forward_rolling']:
            n_iter = \
                (len(self.y) - self.training_window - self.oos_window - self.test_train_diff_days) / \
                self.step_size + 2
            n_iter = int(n_iter)
            last_upper_bound = self.training_window + (n_iter - 1) * self.step_size
            last_oos_size = self.y.shape[0] - last_upper_bound - self.test_train_diff_days

            assert self.X.index.is_monotonic_increasing
            assert self.y.index.is_monotonic_increasing

            for i in range(0, n_iter):

                train_lower_bound = 0 if self.backtest_method == 'walk_forward_extending' else \
                    i * self.step_size
                train_upper_bound = self.training_window + i * self.step_size
                test_lower_bound = train_upper_bound + self.test_train_diff_days
                test_upper_bound = test_lower_bound + self.oos_window
                X_train = self.X.loc[self.unique_dates[train_lower_bound: train_upper_bound]]
                y_train = self.y.loc[self.unique_dates[train_lower_bound: train_upper_bound]]
                X_test = self.X.loc[self.unique_dates[test_lower_bound: test_upper_bound]]
                y_test = self.y.loc[self.unique_dates[test_lower_bound: test_upper_bound]]

                sample_weight_train = None
                if self.sample_weight is not None:
                    sample_weight_train = self.sample_weight[train_lower_bound:train_upper_bound]


                if self.backtest_method == 'walk_forward_rolling':
                    assert (X_train.shape[0] == self.training_window)
                else:
                    assert (X_train.shape[0] == self.training_window + i * self.step_size)
                if i < n_iter - 1:
                    assert (X_test.shape[0] == self.oos_window)
                if i == n_iter - 1:
                    assert (X_test.shape[0] == last_oos_size)

                self._train_clf(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                run_no=i, sample_weight=sample_weight_train)

            if self.training_results['oos_pred'] and self.training_results['oos_truth']:
                self.res_df = pd.DataFrame()
                self.res_df[self.date_column] = self.training_results['oos_index']
                self.res_df['truth'] = self.training_results['oos_truth']
                self.res_df['pred'] = self.training_results['oos_pred']
                self.res_df['run_no'] = self.training_results['run_no']
                self.res_df['day_in_oos_test'] = self.res_df.groupby('run_no').cumcount()
                if 'oos_prob' in self.training_results:
                    self.res_df['prob'] = self.training_results['oos_prob']

    def _calculate_test_results_on_clf(self, X_test, y_test, run_no):
        # calculate probabilities for class and regr
        # in case when no probabilites can be produced by clf, only use the prediction function
        # calc

        # Check if Test Set length is not zero. In the last run of the estimator, this can happen.
        if X_test.shape[0] == 0:
            return

        # this is in both cases regr / class correct

        self.training_results['oos_index'].extend(list(y_test.index))
        self.training_results['oos_truth'].extend(list(y_test))
        self.training_results['run_no'].extend([run_no] * len(y_test))

        # This vector should be filled in the case of classifications
        if hasattr(self.clf, 'predict_proba'):
            self.training_results['oos_prob'].extend(list(self.clf.predict_proba(X_test)[:, 1]))

        # if hasattr(self.clf, 'score'):
        #    utils.lprint(self.logger, self.verbose,
        #                 'Out Of Sample Score in Run %d: %1.4f' % (run_no, self.clf.score(X_test, y_test)))

        # This vector should be filled in the case of classifications and regressions
        if hasattr(self.clf, 'predict'):
            self.training_results['oos_pred'].extend(list(self.clf.predict(X_test)))
            if len(self.training_results['oos_pred']) == 0:
                raise Exception('No elements in oos_pred vector')
        else:
            raise Exception('No predictions can be made with this predictor')

    def _train_clf(self, X_train=None, X_test=None, y_train=None, y_test=None,
                   run_no=1, sample_weight=None):
        # for debugging convenience, store these variables on object level
            # In some cases, X_test will be of length zero in the walk forward method backtest.
            # This case is caught here.

            assert y_train.index.is_monotonic_increasing
            assert y_test.index.is_monotonic_increasing


            if X_test.shape[0] > 0:
                self.last_X_test, self.last_X_train, self.last_y_test, self.last_y_train = \
                    X_test, X_train, y_test, y_train

            if sample_weight is not None and len(sample_weight) != len(y_train):
                print('>>>>Number of Sample Weights does not match the number of Training Samples<<<<')


            if 'sample_weight' in signature(self.clf.fit).parameters.keys():
                self.clf.fit(X=X_train, y= y_train, sample_weight=sample_weight)
            else:
                self.clf.fit(X_train, y_train)

            self._calculate_test_results_on_clf(X_test=X_test, y_test=y_test, run_no=run_no)

            # Store the insample training results
            if self.backtest_method == 'walk_forward_rolling':
                lower_index = -(run_no > 0) * self.step_size
                if not y_train.index.is_monotonic_increasing:
                    raise AttributeError('y_train index is not monotonicall increasing. This should not happen.')

                assert y_train.index[lower_index:].is_monotonic_increasing
                assert pd.Series(self.in_sample_results['idx']).is_monotonic_increasing


                self.in_sample_results['idx'].extend(list(y_train.index[lower_index:]))

                if not pd.Series(self.in_sample_results['idx'][lower_index:]).is_monotonic_increasing:
                    print(y_train.index[lower_index:])
                    raise AttributeError('Last addition to in_sample_results["idx"] rendered no increasing')
                self.in_sample_results['pred'].extend(
                    list(self.clf.predict(X_train[lower_index:])))
                self.in_sample_results['truth'].extend(list(y_train[lower_index:]))
                if hasattr(self.clf, 'predict_proba'):
                    self.in_sample_results['prob'].extend(list(self.clf.predict_proba(X_train[lower_index:])[:, 1]))

            else:
                # Due to the expanding window, this will always overwrite the existing set of pred and truth. Hence,
                # the first score cannot be reproduced on these data
                self.in_sample_results['pred'] = self.clf.predict(X_train)
                self.in_sample_results['truth'] = y_train
                self.in_sample_results['idx'] = list(y_train.index)
                if hasattr(self.clf, 'predict_proba'):
                    self.in_sample_results['prob'] = list(self.clf.predict_proba(X_train)[:, 1])


