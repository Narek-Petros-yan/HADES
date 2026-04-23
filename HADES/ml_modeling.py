import json
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from optuna.exceptions import OptunaError


class HPOptimizer:
	"""
		A class for performing hyperparameter optimization using Optuna.

		The class performs hyperparameter tuning for machine learning models by suggesting
		hyperparameter values from predefined search spaces, evaluating them through cross-validation,
		and stopping early if the best score does not improve for a specified number of iterations.

		Attributes:
			param_spaces (dict): A dictionary defining the hyperparameter search spaces for various models.
			X (pd.DataFrame): The feature matrix for training.
			y (pd.Series or np.array): The target values.
			early_stopping (int): Number of iterations with no improvement before stopping the optimization (default is 2).
			num_cv_folds (int): The number of folds for cross-validation (default is 3).
			param_types (dict): A dictionary mapping parameter types to corresponding Optuna trial methods.
		"""

	def __init__(self, param_spaces, X, y, early_stopping=10, num_cv_folds=3, random_seed=42):
		"""
		Initializes the HPOptimizer with the provided hyperparameter search space and training data.

		Args:
			param_spaces (dict): A dictionary containing the parameter search spaces for models.
				Example:
				param_spaces = {
					RandomForestClassifier: {
						"params": {
							"n_estimators": ("int", 200, 1500, 100),
							"max_features": ("categorical", ['sqrt', 'log2']),
						},
						"extra_args": {"n_jobs": -1, "random_state": 42},
					}
				}
			X (pd.DataFrame): The feature matrix for training.
			y (pd.Series or np.array): The target values for training.
			early_stopping (int, optional): Number of iterations with no improvement before stopping optimization (default is 2).
			num_cv_folds (int, optional): Number of folds for cross-validation (default is 3).
		"""
		self.param_spaces = param_spaces
		self.X = X
		self.y = y
		self.early_stopping = early_stopping
		self.num_cv_folds = num_cv_folds
		self.param_types = {
			"int": lambda trial, name, low, high, step=1: trial.suggest_int(name, low, high, step=step),
			"float": optuna.trial.Trial.suggest_float,
			"loguniform": lambda trial, name, low, high: trial.suggest_float(name, low, high, log=True),
			"categorical": optuna.trial.Trial.suggest_categorical,
		}
		self.random_seed = random_seed

	# source: https://github.com/optuna/optuna/issues/1001#issuecomment-1002821540
	def early_stopping_opt(self, study, trial):
		"""
		Early stopping callback function for Optuna optimization.

		This function raises an `OptunaError` if the study's best trial has not improved
		for a number of consecutive trials specified by `early_stopping`.

		Args:
			study (optuna.Study): The current Optuna study being optimized.
			trial (optuna.Trial): The current trial being evaluated.

		Raises:
			OptunaError: If early stopping criteria are met.
		"""
		if trial.number - study.best_trial.number >= self.early_stopping:
			raise OptunaError()

	def suggest_params(self, trial, model_class):
		"""
		Suggests hyperparameters for a given model class based on predefined search space.

		Args:
			trial (optuna.Trial): The Optuna trial object used to suggest parameters.
			model_class (type): The class of the machine learning model to tune.

		Returns:
			dict: A dictionary of suggested hyperparameters for the model.
		"""
		param_space = self.param_spaces[model_class]["params"]
		return {p: self.param_types[param_type](trial, p, *values) for p, (param_type, *values) in param_space.items()}

	def objective(self, trial, model_class):
		"""
		Objective function to be minimized or maximized by Optuna. It evaluates the model
		with the suggested hyperparameters using cross-validation and returns the average score.

		Args:
			trial (optuna.Trial): The Optuna trial object used to suggest parameters.
			model_class (type): The class of the machine learning model to tune.

		Returns:
			float: The mean cross-validation score for the model with the suggested hyperparameters.
		"""
		params = self.suggest_params(trial, model_class)
		extra_args = self.param_spaces[model_class]["extra_args"]
		model = model_class(**params, **extra_args)
		return cross_val_score(model, self.X, self.y, cv=self.num_cv_folds, scoring="accuracy").mean()

	def param_search(self):
		"""
		Performs hyperparameter search for all models defined in `param_spaces` using Optuna.

		This method creates a study for each model class, performs optimization, and stores the best parameters
		and scores for each model.

		Returns:
			dict: A dictionary containing the best parameters and scores for each model.
		"""
		results = {}

		for model_class in self.param_spaces.keys():
			sampler = TPESampler(seed=self.random_seed)
			study = optuna.create_study(direction="maximize", sampler=sampler)
			try:
				study.optimize(lambda trial: self.objective(trial, model_class), callbacks=[self.early_stopping_opt])
			except OptunaError:
				print(f'EarlyStopping Exceeded: No new best scores for {self.early_stopping} consecutive iterations')
			results[model_class.__name__] = {"best_params": study.best_params, "best_score": study.best_value}
			print(f"Best hyperparameters for {model_class.__name__}:", study.best_params)

		return results


class MLModelEnsemble:

	def __init__(self, opt_params_file):
		self.opt_params_file = opt_params_file
		with open(self.opt_params_file, "r") as f:
			self.opt_params = json.load(f)
		self.models = self.load_all_models()

	def load_all_models(self):
		models = {}
		for model_name, model_params in self.opt_params_file.items():
			models[model_name] = {}
			models[model_name]['model'] = eval(model_name)(**model_params['best_params'])
			models[model_name]['score'] = model_params['best_score']
		return models

	def get_best_model(self):
		max_score = 0
		for model_name, items in self.models.items():
			if items['score'] > max_score:
				max_score = items['score']
				model = items['model']
		return model

	def fit_all(self, X, y):
		for model_name, items in self.models.items():
			items['model'].fit(X, y)

	def predict_proba_ensemble(self, X):
		preds = []
		for model_name, items in self.models.items():
			preds.append(items['model'].predict_proba(X)[:, 1])
		return np.array(preds).mean(axis=1)


if __name__ == "__main__":
	# Define hyperparameter search spaces for different models
	param_spaces = {
		RandomForestClassifier: {
			"params": {
				"n_estimators": ("int", 200, 1500, 100),
				"max_features": ("categorical", ['sqrt', 'log2', 100, 200, 300]),
				"max_depth": ("int", 2, 50, 2),
				"criterion": ("categorical", ['gini', 'entropy', 'log_loss']),
				"min_samples_split": ("int", 2, 20, 2),
				"bootstrap": ("categorical", [True, False]),
			},
			"extra_args": {"n_jobs": -1, "random_state": 42},
		},
		HistGradientBoostingClassifier: {
			"params": {
				"max_iter": ("int", 200, 1500, 100),
				"max_features": ("float", 0.3, 1.0),
				"learning_rate": ("loguniform", 0.01, 0.5),
				"max_depth": ("int", 2, 10, 1),
				"max_bins": ("int", 63, 255, 16),
				"min_samples_leaf": ("int", 1, 101, 2),
			},
			"extra_args": {"random_state": 42},
		},
		XGBClassifier: {
			"params": {
				"n_estimators": ("int", 200, 1500, 100),
				"learning_rate": ("loguniform", 0.01, 0.5),
				"max_depth": ("int", 2, 10, 1),
				"subsample": ("float", 0.6, 1.0),
				"colsample_bytree": ("float", 0.3, 1.0),
			},
			"extra_args": {"device": "cuda", "seed": 42},
		},
		CatBoostClassifier: {
			"params": {
				"iterations": ("int", 100, 5000, 100),
				"learning_rate": ("loguniform", 0.01, 0.5),
				"depth": ("int", 2, 10, 1),
				"l2_leaf_reg": ("loguniform", 1, 5),
				"border_count": ("int", 32, 128, 32),
			},
			"extra_args": {"verbose": 0, "task_type": "GPU", "devices": "0", "random_seed": 42},
		},
		LGBMClassifier: {
			"params": {
				"n_estimators": ("int", 200, 1500, 100),
				"learning_rate": ("loguniform", 0.01, 0.5),
				"max_depth": ("int", 2, 10, 1),
				"num_leaves": ("int", 30, 100, 10),
				"min_data_in_leaf": ("int", 5, 50, 5),
			},
			"extra_args": {"verbose": -1, "n_jobs": -1, "random_state": 42},
		},
	}
	# TODO: making pathing in a proper way
	X_train = pd.read_parquet(r'/home/predator-3/PycharmProjects/QED-private/data/x_train')
	y_train = pd.read_csv(r'/home/predator-3/PycharmProjects/QED-private/data/y_train.csv')
	y_train = np.array(y_train == 4).reshape(-1, )
	optimize = HPOptimizer(param_spaces, X_train, y_train)
	results = optimize.param_search()
	print(results)
	with open("optuna_results_final_old.json", "w") as f:
		json.dump(results, f, indent=4)
