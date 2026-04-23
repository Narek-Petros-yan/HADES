import numpy as np
from sklearn.base import clone
from tqdm.auto import tqdm
from sklearn.model_selection import cross_val_score


class CorrDrop:
	"""
	A class to iteratively drop highly correlated features based on a correlation threshold
	and optimize the threshold using cross-validation.

	Attributes:
		model: A scikit-learn compatible model used for cross-validation.
	"""

	def __init__(self, model):
		"""
		Initializes the CorrDrop class with a given machine learning model.

		Args:
			model: A scikit-learn model instance.
		"""
		self.model = model

	def get_corr_threshold(self, X, y):
		"""
		Finds the optimal correlation threshold that maximizes cross-validation accuracy.

		Iteratively drops highly correlated features at different thresholds and evaluates
		the model performance using cross-validation.

		Args:
			X (pd.DataFrame): Feature matrix.
			y (pd.Series or np.array): Target values.

		Returns:
			dict: A dictionary mapping correlation thresholds to cross-validation scores.
		"""
		corr_data = X.corr()
		corr_data_norm = corr_data - np.eye(len(corr_data))  # Remove self-correlation
		X_drop = X.copy()
		cv_thresh_dict = {}
		cv_thresh_std_dict = {}
		for threshold in tqdm(np.linspace(0.99, 0.01, 99)):
			corr_data_norm, drop_columns = self.drop_corr_columns(corr_data_norm, threshold)
			X_drop.drop(columns=drop_columns, inplace=True)
			# Temporarily rename columns as strings for sklearn compatibility
			og_columns = X_drop.columns
			X_drop.columns = X_drop.columns.astype(str)
			cv_scores = cross_val_score(clone(self.model), X_drop, y, cv=5, scoring='accuracy')
			X_drop.columns = og_columns  # Restore original column names
			cv_thresh_dict[threshold] = np.mean(cv_scores)
			cv_thresh_std_dict[threshold] = np.std(cv_scores)
		return cv_thresh_dict, cv_thresh_std_dict

	@staticmethod
	def drop_corr_columns(corr_data, threshold):
		"""
		Drops columns that are highly correlated beyond the specified threshold.

		Args:
			corr_data (pd.DataFrame): Correlation matrix.
			threshold (float): Correlation threshold for dropping columns.

		Returns:
			tuple: Updated correlation matrix after dropping columns and a list of dropped column names.
		"""
		drop_columns = []
		while True:
			# Count the number of columns each column is highly correlated with
			col_sum = (abs(corr_data) > threshold).sum(axis=1)
			if col_sum.sum() == 0:
				break
			# Identify and drop the column with the highest number of correlations
			drop_column = col_sum.sort_values().index[-1]
			drop_columns.append(drop_column)
			corr_data.drop(columns=drop_column, index=drop_column, inplace=True)
		return corr_data, drop_columns

	def get_cols_to_keep(self, X, cv_thresh_dict, cv_thresh_std_dict):
		"""
		Determines the set of columns to keep based on the optimal correlation threshold.

		Args:
			X (pd.DataFrame): Feature matrix.
			cv_thresh_dict (dict): Dictionary mapping correlation thresholds to cross-validation scores.
			cv_thresh_std_dict (dict): Dictionary mapping correlation thresholds to cross-validation standard deviations.

		Returns:
			list: List of column names to keep in the final dataset.
		"""
		corr_data = X.corr()
		corr_data = corr_data - np.eye(len(corr_data))
		max_acc = max(cv_thresh_dict.values())
		max_acc_threshold = max(th for th, acc in cv_thresh_dict.items() if acc == max_acc)

		# Find the first threshold lower than max_acc_threshold where upper bound drops below max accuracy
		for th, acc in cv_thresh_dict.items():
			if th < max_acc_threshold and (acc + 2 * cv_thresh_std_dict[th]) < max_acc:
				opt_threshold = th
				break
		else:
			opt_threshold = max_acc_threshold  # Default to max_acc_threshold if no better option is found
		_, drop_cols = self.drop_corr_columns(corr_data, opt_threshold)
		# sort mordred classes by their descriptions and other features just by them as they are string
		# makes sure the order of features is the same for reproducibility
		return sorted(list(set(X.columns) - set(drop_cols)),
					  key=lambda x: x if type(x) == str else x.description()), opt_threshold


if __name__ == '__main__':
	from sklearn.ensemble import RandomForestClassifier  # For classification
	import pandas as pd

	X = pd.read_csv(r'initial_data_featurized.csv')
	y = [0 for _ in range(1, 51)] + [1 for _ in range(51, 100)]
	model_test = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
	corrdrop = CorrDrop(model_test)
	cv_thresh_dict_test, cv_thresh_std_dict_test = corrdrop.get_corr_threshold(X, y)
	keep_cols_test, opt_threshold_test = corrdrop.get_cols_to_keep(X, cv_thresh_dict_test, cv_thresh_std_dict_test)
	print('Non correlated columns determined correctly!')
	print(f'Number of columns was reduced from: {X.shape[1]} to {len(keep_cols_test)}')
	print(f'Optimal correlation threshold cutoff is: {opt_threshold_test}')
	X[keep_cols_test].to_csv('initial_data_corrdropped.csv')
