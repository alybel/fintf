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
            specific_model_settings=None):
        if specific_model_settings is not None:
            backtest_settings = specific_model_settings

        tfc_cand = uinutils.see_if_model_exists_and_load_instead(
            df_orig=df, model=model, target=target, hide_columns=hide_columns, date_column=date_column,
            **backtest_settings)
        print('the exact model on these data was found and loaded from file')
        if tfc_cand is not None:
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
