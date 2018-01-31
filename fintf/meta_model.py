from uin_fc_lib import ts_forecasts
from uin_fc_lib import utils as uinutils
import pandas as pd


class MetaModel(object):
    def __init__(self,
                 df=None,
                 target_dict=None,
                 hide_columns=None,
                 backtest_settings=None,
                 date_column=None,
                 model_cascade=None,
                 model_specific_backtest_settings=None
                 ):
        self.df = df.copy()
        if not self.df.index.name == date_column:
            self.df.set_index(date_column, inplace=True)
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

        self.model_specific_backtest_settings = {}
        if model_specific_backtest_settings is not None:
            self.model_specific_backtest_settings = model_specific_backtest_settings

    @staticmethod
    def _run_in_loop(
            model,
            df,
            target,
            hide_columns,
            backtest_settings,
            date_column,
            model_specific_backtest_settings=None):
        if model_specific_backtest_settings is not None:
            backtest_settings = model_specific_backtest_settings

        tfc_cand = uinutils.see_if_model_exists_and_load_instead(
            df_orig=df, model=model, target=target, hide_columns=hide_columns, date_column=date_column,
            **backtest_settings)
        if tfc_cand is not None:
            print('the exact model on these data was found and loaded from file')
            print("return stored model instead")
            return tfc_cand
        tfc = ts_forecasts.TFC(date_column=date_column, df=df)
        tfc.train_model(model=model, target=target, hide_columns=hide_columns, **backtest_settings)
        uinutils.save_model(tfc, tfc.name)
        return tfc

    def run(self):

        for i, layer in enumerate(self.model_cascade):
            print('running layer %d' % i)
            this_layer_results = {}
            for model in layer:
                specific_model_settings = None
                if model in self.model_specific_backtest_settings and self.model_specific_backtest_settings[model] is not None:
                    specific_model_settings = self.model_specific_backtest_settings[model]
                target_col = None
                for key in self.target_dict:
                    if model in self.target_dict[key]:
                        target_col = key
                print('####')
                print('running model %s' % model)
                print('####')
                tfc = self._run_in_loop(
                    model=layer[model],
                    df=self.df.copy(),
                    hide_columns=self.hide_columns,
                    backtest_settings=self.backtest_settings,
                    target=target_col,
                    date_column=self.date_column,
                    model_specific_backtest_settings=specific_model_settings
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
                    if self.df['%s_%s_prob' % (trained_model, target_col)].isnull().all():
                        raise Exception('joining model results from %s failed' % trained_model)

                # This is the Regression Case
                else:
                    data.rename(columns={'pred': '%s_%s_pred' % (trained_model, target_col)}, inplace=True)
                    self.df = self.df.join(data['%s_%s_pred' % (trained_model, target_col)])
                    if self.df['%s_%s_pred' % (trained_model, target_col)].isnull().all():
                        raise Exception('joining model results from %s failed' % trained_model)

                # clean self.df
                self.df.dropna(inplace=True)
                if self.df.shape[0] == 0:
                    raise Exception('Something went wrong when joining the probabilites from previous models. probably '
                                    'the prediction column is all zero.')

                self.df = self.df.join(data['__oos__'])
                self.df = self.df[self.df['__oos__'] == 1]
                self.df.drop(['__oos__'], axis=1, inplace=True)

    def get_results(self):
        return self.results

    def get_data(self):
        return self.df


from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def meta_model_1(
        df=None,
        class_target='',
        regr_target='',
        vola_target='',
        step_size=100,
        hide_columns=None,
        date_column=None,
        target_horizon=1,
        training_window=1000,
):
    mc = [
        {'m1': Ridge(alpha=5),
         'm2': LogisticRegression(),
         'mv1': RandomForestRegressor(n_estimators=1000, min_samples_leaf=.15),
         'mv2': Ridge(alpha=5),

         },
        {
            'm3': RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_leaf=.15),
            'm4': RandomForestRegressor(n_estimators=1000, max_depth=3, min_samples_leaf=.15),
            'mv4': RandomForestRegressor(n_estimators=1000, max_depth=3, min_samples_leaf=.15),
        },
        {
            'm51': Ridge(alpha=3),
            'm5': Ridge(alpha=5),
            'm6': LogisticRegression(),
            'm63': RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_leaf=.15),
            'm64': RandomForestRegressor(n_estimators=1000, max_depth=3, min_samples_leaf=.15)
        }
    ]

    target_dict = {
        class_target: ['m2', 'm3', 'm6', 'm63'],
        regr_target: ['m1', 'm4', 'm5', 'm51', 'm64'],
        vola_target: ['mv1', 'mv2', 'mv4']
    }

    bs = {
        'backtest_method': 'walk_forward_rolling',
        'training_window': training_window,
        'step_size': step_size,
        'test_train_diff_days': target_horizon
    }

    base = bs.copy()
    base.update({'run_slim': False})
    sms = {'m64': base}

    mm = MetaModel(
        backtest_settings=bs,
        df=df,
        target_dict=target_dict,
        model_cascade=mc,
        date_column=date_column,
        hide_columns=hide_columns,
        model_specific_backtest_settings=sms
    )

    return mm
