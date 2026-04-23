# TODO: Should include th calculation of all features mordred, admet ai, QED, as well as removing invalid features

import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
import pickle
from rdkit.Chem.QED import properties as calc_qed_properties
from mordred import Calculator
from mordred import descriptors
from admet_ai import ADMETModel

# property names of QED
# 'MW" is removed as it is a deuplicate with mordred's version
QED_FEATURE_NAMES = ['ALOGP', 'HBA', 'HBD', 'PSA', 'ROTB', 'AROM', 'ALERTS']
# 'molecular_weight' and 'Lipinski' are removed as they are duplicates with QED and mordred features
# property names of ADMET AI
ADMET_AI_FEATURES = ['logP', 'hydrogen_bond_acceptors', 'hydrogen_bond_donors',
					 'stereo_centers', 'tpsa', 'AMES', 'BBB_Martins', 'Bioavailability_Ma', 'CYP1A2_Veith',
					 'CYP2C19_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2C9_Veith',
					 'CYP2D6_Substrate_CarbonMangels', 'CYP2D6_Veith', 'CYP3A4_Substrate_CarbonMangels', 'CYP3A4_Veith',
					 'Carcinogens_Lagunin', 'ClinTox', 'DILI', 'HIA_Hou', 'NR-AR-LBD', 'NR-AR', 'NR-AhR',
					 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma', 'PAMPA_NCATS', 'Pgp_Broccatelli', 'SR-ARE',
					 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'Skin_Reaction', 'hERG', 'Caco2_Wang',
					 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'Half_Life_Obach',
					 'HydrationFreeEnergy_FreeSolv', 'LD50_Zhu', 'Lipophilicity_AstraZeneca', 'PPBR_AZ',
					 'Solubility_AqSolDB', 'VDss_Lombardo']


def neutralize_atoms(mol):
	# TODO: utils?
	"""Neutralize charges, from rdkit.org/docs/Cookbook.html#neutralizing-molecules"""
	pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
	at_matches = mol.GetSubstructMatches(pattern)
	at_matches_list = [y[0] for y in at_matches]
	if len(at_matches_list) > 0:
		for at_idx in at_matches_list:
			atom = mol.GetAtomWithIdx(at_idx)
			chg = atom.GetFormalCharge()
			hcount = atom.GetTotalNumHs()
			atom.SetFormalCharge(0)
			atom.SetNumExplicitHs(hcount - chg)
			atom.UpdatePropertyCache()
	return mol


class Featurizer:
	"""
	A class to compute molecular features using Mordred descriptors, ADMET properties, and QED features.

	Attributes:
	    admet_featurizer (ADMETModel): An instance of ADMETModel for computing ADMET-related features.
	    mordred_featurizer (Calculator): A Mordred descriptor calculator initialized with selected descriptors.
	    selected_admet (list): A list of selected ADMET features to be computed.
	    selected_qed (list): A list of selected QED features to be computed.
	"""

	def __init__(self, selected_mordred=descriptors, selected_admet=None, selected_qed=None, features_pkl=None):
		"""
		Initializes the Featurizer with specified molecular descriptors, ADMET properties, and QED features.

		If a pickle file (`features_pkl`) is provided, it will override any manually specified `selected_mordred`,
		`selected_admet`, and `selected_qed` values by loading them from the file.

		Args:
		    selected_mordred (list, optional): A list of selected Mordred descriptors. Defaults to `descriptors`.
		    selected_admet (list, optional): A list of selected ADMET properties. Defaults to None.
		    selected_qed (list, optional): A list of selected QED features. Defaults to None.
		    features_pkl (str, optional): Path to a pickle file containing preselected features. Defaults to None.
		"""
		# TODO: reconstruct methods to consider where admet ai or qed features are not used
		if features_pkl:
			selected_mordred, selected_admet, selected_qed = self.load_features(features_pkl)

		self.mordred_featurizer = Calculator(selected_mordred, ignore_3D=True)

		if not selected_admet:
			self.selected_admet = ADMET_AI_FEATURES
		else:
			self.selected_admet = selected_admet
		self.admet_featurizer = ADMETModel()

		if not selected_qed:
			self.selected_qed = QED_FEATURE_NAMES
		else:
			self.selected_qed = selected_qed

	@staticmethod
	def load_features(pkl_file):
		"""
		Loads preselected features from a pickle file and categorizes them into Mordred, ADMET, and QED features.

		Args:
		    pkl_file (str): Path to the pickle file containing feature names.

		Returns:
		    tuple: A tuple containing three lists - (mordred_features, admet_features, qed_features).
		"""
		with open(pkl_file, 'rb') as f:
			features = pickle.load(f)
		mordred_feats, admet_feats, qed_feats = [], [], []
		for col in features:
			if 'mordred' in col.__repr__():
				mordred_feats.append(col)
			elif col in QED_FEATURE_NAMES:
				qed_feats.append(col)
			else:
				admet_feats.append(col)
		return mordred_feats, admet_feats, qed_feats

	@staticmethod
	def standartize_mol(mol):
		# TODO: utils?
		"""
		Standardizes a given molecule by removing salts, neutralizing atoms, and removing stereochemistry.

		Args:
		    mol (rdkit.Chem.rdchem.Mol): The input molecule.

		Returns:
		    rdkit.Chem.rdchem.Mol: The standardized molecule.
		"""
		remover = SaltRemover.SaltRemover()
		stripped_mol = remover.StripMol(mol)
		neutralized_mol = neutralize_atoms(stripped_mol)
		Chem.RemoveStereochemistry(neutralized_mol)
		return neutralized_mol

	def featurize_smiles(self, smiles):
		"""
		Computes features for a given SMILES string, including Mordred, ADMET AI, and QED features.

		Args:
		    smiles (str): A SMILES representation of the molecule.

		Returns:
		    list: A list of computed features concatenated from Mordred, ADMET AI, and QED calculations.
		"""

		mol = Chem.MolFromSmiles(smiles)
		if mol is None:
			# Return an empty list if SMILES is invalid
			print(f"Warning: Invalid SMILES string '{smiles}'")
			return []
		mol = self.standartize_mol(mol)
		smiles = Chem.MolToSmiles(mol)

		mordred_feature = list(self.mordred_featurizer(mol).values())

		admet_all_feature = self.admet_featurizer.predict(smiles)
		admet_feature = [admet_all_feature[feature_name] for feature_name in self.selected_admet]

		qed_all_feature = calc_qed_properties(mol)._asdict()
		qed_feature = [qed_all_feature[feature_name] for feature_name in self.selected_qed]

		return mordred_feature + admet_feature + qed_feature

	def featurize_many_smiles(self, smileses_list):
		"""
		Computes features for multiple SMILES strings and returns them as a pandas DataFrame.

		Args:
		    smileses_list (list of str): A list of SMILES strings representing molecules.

		Returns:
		    pandas.DataFrame: A DataFrame containing computed features from Mordred, ADMET AI, and QED calculations.
		"""

		valid_smileses_list, mols_list = [], []
		for smiles in smileses_list:
			mol = Chem.MolFromSmiles(smiles)
			if not mol:
				print(f"Warning: Invalid SMILES string '{smiles}'")
				continue
			mol = self.standartize_mol(mol)
			mols_list.append(mol)
			smiles = Chem.MolToSmiles(mol)
			valid_smileses_list.append(smiles)

		mordred_features = pd.DataFrame([dict(r) for r in self.mordred_featurizer.map(mols_list)])

		admet_features = self.admet_featurizer.predict(valid_smileses_list)
		admet_features = admet_features[self.selected_admet].reset_index(drop=True)

		qed_features = pd.DataFrame([calc_qed_properties(mol)._asdict() for mol in mols_list])
		qed_features = qed_features[self.selected_qed]

		return pd.concat([mordred_features, admet_features, qed_features], axis=1)

	@staticmethod
	def drop_na_cols(data):
		return data[data.dtypes[data.dtypes != object].index]

	@staticmethod
	def drop_same_val_cols(data, thresh=0.75):
		keep_cols = []
		for c, col in enumerate(data.columns):
			val_counts = data.iloc[:, c].value_counts(normalize=True)
			if val_counts.iloc[0] <= thresh:
				keep_cols.append(col)
		return data[keep_cols]


if __name__ == '__main__':
	featurizer = Featurizer()
	print('model_loaded')
	data = [i * 'C' for i in range(1, 100)]
	data_featurized = featurizer.featurize_many_smiles(data)
	na_dropped_data = featurizer.drop_na_cols(data_featurized)
	same_val_dropped_data = featurizer.drop_same_val_cols(na_dropped_data)
	print('Features created succesfully!')
	with open('initial_columns', 'wb') as f:
		pickle.dump(list(same_val_dropped_data.columns), f)
	na_dropped_data.to_csv('initial_data_featurized.csv')
