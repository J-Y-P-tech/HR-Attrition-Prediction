import warnings
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    SGDClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier

# Third-party gradient boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


class ModelTrainer:
    """
    A class to train, evaluate, and tune a suite of classification models.
    """

    def __init__(self, resampling_strategy: str = "smote", random_state: int = 42):
        """
        Initializes the ModelTrainer.
        Args:
        resampling_strategy (str): One of "smote", "undersample", "smote_tomek", or None.
        random_state (int): The random state for reproducibility.
        """
        self.random_state = random_state
        self.resampler = self._get_resampler(resampling_strategy)
        self.models = self._initialize_models()
        self.results_df = None
        self.best_params_ = {}

    def _get_resampler(self, strategy):
        if strategy == "smote":
            return SMOTE(random_state=self.random_state)
        elif strategy == "undersample":
            return RandomUnderSampler(random_state=self.random_state)
        elif strategy == "smote_tomek":
            return SMOTETomek(random_state=self.random_state)
        else:
            return None

    def _initialize_models(self):
        """Instantiate pipelines for each classifier with optional resampling."""
        clfs = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "GradientBoosting": GradientBoostingClassifier(random_state=self.random_state),
            "AdaBoost": AdaBoostClassifier(random_state=self.random_state),
            "ExtraTrees": ExtraTreesClassifier(random_state=self.random_state),
            "SVC": SVC(probability=True, random_state=self.random_state),
            # Wrap models that don't have predict_proba in CalibratedClassifierCV
            "LinearSVC": CalibratedClassifierCV(LinearSVC(random_state=self.random_state, max_iter=2000, dual="auto")),
            "PassiveAggressive": CalibratedClassifierCV(PassiveAggressiveClassifier(random_state=self.random_state)),
            "KNN": KNeighborsClassifier(),
            "GaussianNB": GaussianNB(),
            "BernoulliNB": BernoulliNB(),
            "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            "QDA": QuadraticDiscriminantAnalysis(reg_param=0.05),
            "SGD": SGDClassifier(max_iter=1000, tol=1e-3, random_state=self.random_state),
            "Ridge": RidgeClassifier(random_state=self.random_state),
            "MLP": MLPClassifier(max_iter=500, random_state=self.random_state),
            "Bagging": BaggingClassifier(random_state=self.random_state),
        }

        if XGBOOST_AVAILABLE:
            clfs["XGBoost"] = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss', use_label_encoder=False)
        if LIGHTGBM_AVAILABLE:
            clfs["LightGBM"] = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        if CATBOOST_AVAILABLE:
            clfs["CatBoost"] = cb.CatBoostClassifier(random_state=self.random_state, verbose=False)

        # Define ensemble models using already defined base classifiers
        voting_estimators = [("lr", clfs["LogisticRegression"]), ("rf", clfs["RandomForest"]), ("svc", clfs["SVC"])]
        clfs["Voting"] = VotingClassifier(estimators=voting_estimators, voting="soft")

        stacking_estimators = [("dt", clfs["DecisionTree"]), ("knn", clfs["KNN"]), ("nb", clfs["GaussianNB"])]
        clfs["Stacking"] = StackingClassifier(estimators=stacking_estimators, final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state), passthrough=False)

        pipelines = {}
        for name, clf in clfs.items():
            steps = []
            if self.resampler is not None:
                steps.append(("resample", self.resampler))
            steps.append(("clf", clf))
            pipelines[name] = ImbPipeline(steps=steps)
        return pipelines

    def _evaluate_model(self, name, y_test, y_pred, y_scores):
        """Calculates all metrics and returns a result dictionary."""
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
        
        cm_raw = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm_raw.ravel() if cm_raw.size == 4 else (0, 0, 0, 0)
        total = cm_raw.sum() if cm_raw.sum() > 0 else 1

        return {
            "model": name, "precision": precision, "recall": recall, "f1_score": f1,
            "roc_auc": roc_auc, "pr_auc": pr_auc, "True_negative": (tn / total) * 100,
            "False_negative": (fn / total) * 100, "True_positive": (tp / total) * 100,
            "False_positive": (fp / total) * 100,
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train each model with default parameters and evaluate on the test set.
        Returns a DataFrame with performance metrics for each model.
        """
        records = []
        for name, pipeline in self.models.items():
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                # Get probability scores for ROC AUC calculation
                if hasattr(pipeline.named_steps["clf"], "predict_proba"):
                    y_scores = pipeline.predict_proba(X_test)[:, 1]
                elif hasattr(pipeline.named_steps["clf"], "decision_function"):
                    y_scores = pipeline.decision_function(X_test)
                else:
                    # Fallback for classifiers without native score output
                    y_scores = y_pred.astype(float)

                record = self._evaluate_model(name, y_test, y_pred, y_scores)
                records.append(record)
            except Exception:
                # Silently skip any model that fails during default training
                continue
        
        df = pd.DataFrame(records).set_index("model").round(3)
        self.results_df = df
        return df

    def _get_param_distributions(self):
        """Returns a dictionary of hyperparameter search spaces for each model."""
        return {
            "LogisticRegression": {'clf__penalty': ['l1', 'l2', 'elasticnet'], 'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__solver': ['saga'], 'clf__l1_ratio': [0.3, 0.5, 0.7], 'clf__max_iter': [2000]},
            "RandomForest": {'clf__n_estimators': [100, 200, 500], 'clf__max_depth': [10, 20, 30, None], 'clf__min_samples_split': [2, 5, 10], 'clf__min_samples_leaf': [1, 2, 4]},
            "ExtraTrees": {'clf__n_estimators': [100, 200, 500], 'clf__max_depth': [10, 20, 30, None], 'clf__max_features': ['sqrt', 'log2', None], 'clf__min_samples_split': [2, 5, 10], 'clf__min_samples_leaf': [1, 2, 4], 'clf__bootstrap': [True, False]},
            "GradientBoosting": {'clf__n_estimators': [100, 200, 500], 'clf__learning_rate': [0.01, 0.1, 0.2], 'clf__max_depth': [3, 5, 8], 'clf__subsample': [0.8, 1.0]},
            "AdaBoost": {'clf__n_estimators': [50, 100, 200], 'clf__learning_rate': [0.01, 0.1, 1.0]},
            "SVC": {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto'], 'clf__kernel': ['rbf', 'poly']},
            "MLP": {'clf__hidden_layer_sizes': [(50, 50), (100,), (100, 50, 25)], 'clf__activation': ['relu', 'tanh'], 'clf__alpha': [0.0001, 0.001, 0.01], 'clf__learning_rate_init': [0.001, 0.01]},
            "KNN": {'clf__n_neighbors': [3, 5, 7, 9], 'clf__weights': ['uniform', 'distance'], 'clf__p': [1, 2]},
            "XGBoost": {'clf__n_estimators': [100, 200, 500], 'clf__learning_rate': [0.01, 0.1, 0.2], 'clf__max_depth': [3, 5, 7], 'clf__subsample': [0.8, 1.0], 'clf__colsample_bytree': [0.8, 1.0]},
            "LightGBM": {'clf__n_estimators': [100, 200, 500], 'clf__learning_rate': [0.01, 0.05, 0.1], 'clf__num_leaves': [20, 31, 40], 'clf__max_depth': [-1, 5, 10]},
            "CatBoost": {'clf__iterations': [200, 500], 'clf__learning_rate': [0.01, 0.1], 'clf__depth': [4, 6, 8]},
            # For calibrated models, target the 'estimator' parameter of CalibratedClassifierCV
            "LinearSVC": {'clf__estimator__C': [0.01, 0.1, 1, 10, 100], 'clf__estimator__loss': ['hinge', 'squared_hinge'], 'clf__estimator__max_iter': [2000, 5000]},
            "PassiveAggressive": {'clf__estimator__C': [0.01, 0.1, 0.5, 1.0], 'clf__estimator__loss': ['hinge', 'squared_hinge'], 'clf__estimator__max_iter': [1000, 2000]},
        }

    
    def tune_and_evaluate(self, X_train, X_test, y_train, y_test, n_iter=50, scoring='roc_auc'):
        """
        Tunes hyperparameters using RandomizedSearchCV, evaluates the best models,
        and caches both the best parameters and the fitted estimators for later use.
        
        Args:
            n_iter (int): Number of parameter settings that are sampled.
            scoring (str): The metric to optimize during tuning (e.g., 'roc_auc', 'f1').

        Returns:
            A DataFrame with performance metrics and the best hyperparameters for each tuned model.
        """
        records = []
        param_distributions = self._get_param_distributions()
        
        # Clear previous results before a new run
        self.best_params_ = {}
        self.best_estimators_ = {}

        for name, pipeline in self.models.items():
            if name not in param_distributions:
                continue  # Silently skip models without a parameter distribution

            try:
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_distributions[name],
                    n_iter=n_iter,
                    cv=5,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0  # Suppress fitting messages
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    search.fit(X_train, y_train)
                
                # --- START OF NEW CODE ---

                # Get the best estimator (it's already been refitted on the full training data)
                best_model = search.best_estimator_
                
                # Get the dictionary of best parameters
                best_params = search.best_params_

                # Store the fitted model and its parameters in our dictionaries
                self.best_estimators_[name] = best_model
                self.best_params_[name] = best_params
                
                # --- END OF NEW CODE ---

                # Now, evaluate this best model on the test set
                y_pred = best_model.predict(X_test)
                y_scores = best_model.predict_proba(X_test)[:, 1]

                record = self._evaluate_model(name, y_test, y_pred, y_scores)
                
                # Add the best parameters to the results record for easy viewing in the DataFrame
                record['best_params'] = best_params
                records.append(record)

            except Exception:
                # Silently skip any model that fails during tuning
                continue
        
        df = pd.DataFrame(records).set_index("model").sort_values(by=scoring, ascending=False).round(3)
        self.results_df = df
        return df