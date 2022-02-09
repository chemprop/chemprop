import random

import numpy as np
# import GPy
# import heapq
from argparse import Namespace
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from chemprop.utils import negative_log_likelihood


def uncertainty_estimator_builder(uncertainty_method: str):
    return {
        'nn': NNEstimator,
        # 'gaussian': GaussianProcessEstimator,
        'random_forest': RandomForestEstimator,
        # 'tanimoto': TanimotoEstimator,
        'ensemble': EnsembleEstimator,
        # 'latent_space': LatentSpaceEstimator,
        # 'bootstrap': BootstrapEstimator,
        # 'snapshot': SnapshotEstimator,
        # 'dropout': DropoutEstimator,
        # 'fp_random_forest': FPRandomForestEstimator,
        # 'fp_gaussian': FPGaussianProcessEstimator,
        'evidence': EvidenceEstimator,
        'sigmoid': SigmoidEstimator,
    }[uncertainty_method]


class UncertaintyEstimator:
    def __init__(self, train_data, new_data, scaler, args):
        self.train_data = train_data
        self.new_data = new_data
        self.unc = np.zeros(shape=(len(self.new_data.smiles()), args.num_tasks))
        self.var = np.zeros(shape=(len(self.new_data.smiles()), args.num_tasks))
        self.computed_std = None
        self.scaler = scaler
        self.args = args

    def process_model(self, model, predict):
        pass

    def compute_uncertainty(self, new_preds):
        pass

    def _scale_uncertainty(self, uncertainty):
        return self.scaler.stds * uncertainty

    def export_std(self): 
        """ Export the computed std function. This is handeled in compute_uncertainty"""
        return self.computed_std


class UnionEstimator(UncertaintyEstimator):
    def __init__(self, train_data, new_data, scaler, args):
        super().__init__(train_data, new_data, scaler, args)

        self.sum_last_hidden_train = np.zeros(
            (len(self.train_data.smiles()), self.args.last_hidden_size))
        self.sum_last_hidden_new = np.zeros(
            (len(self.new_data.smiles()), self.args.last_hidden_size))

    def process_model(self, model, predict):
        model.eval()
        model.use_last_hidden = False

        last_hidden_train = predict(
            model=model,
            data=self.train_data,
            batch_size=self.args.batch_size,
            scaler=None,
            quiet=self.args.quiet
        )
        self.sum_last_hidden_train += np.array(last_hidden_train)

        last_hidden_new = predict(
            model=model,
            data=self.new_data,
            batch_size=self.args.batch_size,
            scaler=None,
            quiet=self.args.quiet
        )
        self.sum_last_hidden_new += np.array(last_hidden_new)

    def _compute_hidden_vals(self):
        avg_last_hidden_train = self.sum_last_hidden_train / self.args.ensemble_size
        avg_last_hidden_new = self.sum_last_hidden_new / self.args.ensemble_size
        return avg_last_hidden_train, avg_last_hidden_new


class NNEstimator(UncertaintyEstimator):
    def __init__(self, train_data, new_data, scaler, args):
        super().__init__(train_data, new_data, scaler, args)

    def process_model(self, model, predict):
        preds, uncertainty = predict(
            model=model,
            data=self.new_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            quiet=self.args.quiet,
            uncertainty=True,
        )
        if len(preds) != 0:
            self.unc += np.array(uncertainty).clip(min=0)

    def compute_uncertainty(self, new_preds):
        return (new_preds, np.sqrt(self.unc / self.args.ensemble_size))


class SigmoidEstimator(UncertaintyEstimator):
    def __init__(self, train_data, new_data, scaler, args):
        super().__init__(train_data, new_data, scaler, args)

    def process_model(self, model, predict):
        def categorical_variance(p): #assumes binary classification
            return p*(1-p)**2  + (1-p)*p**2

        preds = predict(
            model=model,
            data=self.new_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            quiet=self.args.quiet,
            uncertainty=True,
        )

        # Calculate the variance (option to compute entropy in run_training)
        var = categorical_variance(np.array(preds))
        if len(preds) != 0:
            self.unc += var.clip(min=0)

    def compute_uncertainty(self, new_preds):
        return (new_preds, np.sqrt(self.unc / self.args.ensemble_size))


class EvidenceEstimator(UncertaintyEstimator):
    def __init__(self, train_data, new_data, scaler, args):
        super().__init__(train_data, new_data, scaler, args)

    def process_model(self, model, predict):
        preds, uncertainty, var  = predict(
            model=model,
            data=self.new_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            quiet=self.args.quiet,
            uncertainty=True, 
            export_var = True, 
        )

        if len(preds) != 0:
            self.unc += np.array(uncertainty).clip(min=0)
            self.var += np.array(var).clip(min=0)

    def compute_uncertainty(self, new_preds):
        # Compute std for use in calibration curves
        self.computed_std = np.sqrt(self.var / self.args.ensemble_size)
        return (new_preds, np.sqrt(self.unc / self.args.ensemble_size))


# class GaussianProcessEstimator(DroppingEstimator):
#     def compute_confidence(self, new_preds):
#         avg_last_hidden_train, avg_last_hidden_new = self._compute_hidden_vals()
#         new_preds = np.zeros_like(self.conf)

#         transformed = self.scaler.transform(np.array(self.train_data.targets()))

#         for task in range(self.args.num_tasks):
#             kernel = GPy.kern.Linear(input_dim=self.args.last_hidden_size)
#             gaussian = GPy.models.SparseGPRegression(
#                 avg_last_hidden_train,
#                 transformed[:, task:task + 1], kernel)
#             gaussian.optimize()

#             avg_preds, avg_var = gaussian.predict(avg_last_hidden_new)

#             new_preds[:, task:task+1] = avg_preds
#             self.conf[:, task:task+1] = np.sqrt(avg_var)

#         return (self.scaler.inverse_transform(new_preds),
#                 self._scale_confidence(self.conf))


# class FPGaussianProcessEstimator(ConfidenceEstimator):
#     def compute_confidence(self, new_preds):
#         train_smiles = self.train_data.smiles()
#         new_smiles = self.new_data.smiles()
#         train_fps = np.array([morgan_fingerprint(s) for s in train_smiles])
#         new_fps = np.array([morgan_fingerprint(s) for s in new_smiles])

#         new_preds = np.zeros_like(self.conf)

#         # Train targets are already scaled.
#         scaled_train_targets = np.array(self.train_data.targets())

#         for task in range(self.args.num_tasks):
#             kernel = GPy.kern.Linear(input_dim=train_fps.shape[1])
#             gaussian = GPy.models.SparseGPRegression(
#                 train_fps,
#                 scaled_train_targets[:, task:task + 1], kernel)
#             gaussian.optimize()

#             avg_preds, avg_var = gaussian.predict(new_fps)

#             new_preds[:, task:task+1] = avg_preds
#             self.conf[:, task:task+1] = np.sqrt(avg_var)

#         return (self.scaler.inverse_transform(new_preds),
#             self._scale_confidence(self.conf))


class RandomForestEstimator(UnionEstimator):
    def compute_uncertainty(self, new_preds):
        avg_last_hidden_train, avg_last_hidden_new = self._compute_hidden_vals()
        transformed = self.scaler.transform(np.array(self.train_data.targets()))

        new_preds = np.zeros_like(self.train_unc)

        n_trees = 128
        for task in range(self.args.num_tasks):
            forest = RandomForestRegressor(n_estimators=n_trees)
            forest.fit(avg_last_hidden_train, transformed[:, task])

            new_preds[:, task] = forest.predict(avg_last_hidden_new)
            individual_predictions = np.array([
                estimator.predict(avg_last_hidden_new) for estimator in forest.estimators_])
            self.unc[:, task] = np.std(individual_predictions, axis=0)

        return (self.scaler.inverse_transform(new_preds),
                self._scale_uncertainty(self.unc))


# class FPRandomForestEstimator(ConfidenceEstimator):
#     def compute_confidence(self, new_preds):
#         train_smiles = self.train_data.smiles()
#         new_smiles = self.train_data.smiles()
#         train_fps = np.array([morgan_fingerprint(s) for s in train_smiles])
#         new_fps = np.array([morgan_fingerprint(s) for s in new_smiles])

#         new_preds = np.zeros_like(self.conf)

#         # Train targets are already scaled.
#         scaled_train_targets = np.array(self.train_data.targets())

#         n_trees = 128
#         for task in range(self.args.num_tasks):
#             forest = RandomForestRegressor(n_estimators=n_trees)
#             forest.fit(train_fps, scaled_train_targets[:, task])

#             new_preds[:, task] = forest.predict(new_fps)
#             individual_predictions = np.array([
#                 estimator.predict(new_fps) for estimator in forest.estimators_])
#             self.conf[:, task] = np.std(individual_predictions, axis=0)

#         return (self.scaler.inverse_transform(new_preds),
#                 self._scale_confidence(self.conf))


# class LatentSpaceEstimator(DroppingEstimator):
#     def compute_confidence(self, new_preds):
#         avg_last_hidden_train, avg_last_hidden_new = self._compute_hidden_vals()

#         for input_ in range(len(avg_last_hidden_new)):
#             distances = np.zeros(len(avg_last_hidden_train))
#             for train_input in range(len(avg_last_hidden_train)):
#                 difference = avg_last_hidden_new[input_] - avg_last_hidden_train[train_input]
#                 distances[train_input] = np.sqrt(np.sum(difference * difference))

#             self.conf[input_, :] = sum(heapq.nsmallest(5, distances))/5

#         return (new_preds, self.conf)


class EnsembleEstimator(UncertaintyEstimator):
    def __init__(self, train_data, new_data, scaler, args):
        super().__init__(train_data, new_data, scaler, args)
        self.all_preds = []

    def process_model(self, model, predict):
        preds = predict(
            model=model,
            data=self.new_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            quiet=self.args.quiet,
        )
        self.all_preds.append(preds)

    def compute_uncertainty(self, new_preds):
        return (new_preds, np.std(self.all_preds, axis=0))


# class BootstrapEstimator(EnsembleEstimator):
#     def __init__(self, train_data, new_data, scaler, args):
#         super().__init__(train_data, new_data, scaler, args)


# class SnapshotEstimator(EnsembleEstimator):
#     def __init__(self, train_data, new_data, scaler, args):
#         super().__init__(train_data, new_data, scaler, args)


# class DropoutEstimator(EnsembleEstimator):
#     def __init__(self, train_data, new_data, scaler, args):
#         super().__init__(train_data, new_data, scaler, args)


# class TanimotoEstimator(ConfidenceEstimator):
#     def compute_confidence(self, new_preds):
#         train_smiles = self.train_data.smiles()
#         new_smiles = self.new_data.smiles()

#         train_smiles_sfp = [morgan_fingerprint(s) for s in train_smiles]
#         new_smiles_sfp = [morgan_fingerprint(s) for s in new_smiles]

#         for i in range(len(new_smiles)):
#             self.conf[i, :] = np.ones((self.args.num_tasks)) * tanimoto(
#                 new_smiles[i], train_smiles_sfp, lambda x: sum(heapq.nsmallest(8, x))/8)

#         return (new_preds, self.conf)


# Classification methods.
# class ConformalEstimator(DroppingEstimator):
#     pass


# class BoostEstimator(DroppingEstimator):
#     pass


# def morgan_fingerprint(smiles: str, radius: int = 3, num_bits: int = 2048,
#                        use_counts: bool = False) -> np.ndarray:
#     """
#     Generates a morgan fingerprint for a smiles string.

#     :param smiles: A smiles string for a molecule.
#     :param radius: The radius of the fingerprint.
#     :param num_bits: The number of bits to use in the fingerprint.
#     :param use_counts: Whether to use counts or just a bit vector for the fingerprint
#     :return: A 1-D numpy array containing the morgan fingerprint.
#     """
#     if type(smiles) == str:
#         mol = Chem.MolFromSmiles(smiles)
#     else:
#         mol = smiles
#     if use_counts:
#         fp_vect = AllChem.GetHashedMorganFingerprint(
#             mol, radius, nBits=num_bits, useChirality=True)
#     else:
#         fp_vect = AllChem.GetMorganFingerprintAsBitVect(
#             mol, radius, nBits=num_bits, useChirality=True)
#     fp = np.zeros((1,))
#     DataStructs.ConvertToNumpyArray(fp_vect, fp)

#     return fp


# def tanimoto(smile, train_smiles_sfp, operation):
#     smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
#     fp = morgan_fingerprint(smiles)
#     tanimoto_distance = []

#     for sfp in train_smiles_sfp:
#         tsim = np.dot(fp, sfp) / (fp.sum() +
#                                   sfp.sum() - np.dot(fp, sfp))
#         tanimoto_distance.append(-np.log2(max(0.0001, tsim)))

#     return operation(tanimoto_distance)
