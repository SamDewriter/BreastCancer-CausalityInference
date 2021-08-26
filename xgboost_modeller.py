import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV 
from mlflow.xgboost import autolog
import logging
import warnings
warnings.filterwarnings('ignore')

class XgModeller():
    def __init__(self) -> None:
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info('XgModeller initialized...')
        self.model = xgb.XGBClassifier(
                                    learning_rate =0.1, n_estimators=10, max_depth=5,
                                    min_child_weight=1, gamma=0, subsample=0.8, 
                                    colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=42
                                    )
        testing_params = {
                        'colsample_bytree':[0.4,0.6,0.8], 'gamma':[0,0.03,0.1,0.3], 
                        'min_child_weight':[1.5,6,10],'learning_rate':[0.1,0.07],
                        'max_depth':[3,5], 'n_estimators':[20], 'subsample':[0.6,0.95], 
                        'reg_alpha':[1e-5, 1e-2,  0.75],'reg_lambda':[1e-5, 1e-2, 0.45]
                        }

        self.testing_params = testing_params
    def get_base_model(self):
        return self.model

    def gridsearch_model(self, X, Y, output=False):
        testing_params = self.testing_params
        base_model = self.model
        logging.info("Randomized+SearchCV in process, 'n_estimators'=15 ...")
        rand_search = RandomizedSearchCV(estimator=base_model, param_distributions=testing_params, 
                                        cv=5, n_iter=10, 
                                        return_train_score=True, n_jobs=1, verbose=2)
        rand_search.fit(X=X, y=Y)
        logging.info('RandomizedSearchCV in completed')
        best_model = rand_search.best_estimator_
        self.best_model = best_model
        logging.info('Best_estimator in retrieved')
        if output:
            return base_model, best_model

    def best_feature_imp(self, max_feats=None):
        logging.info('Feature importance plotting in process...')
        best_model = self.best_model
        xgb.plot_importance(best_model, max_num_features=max_feats)