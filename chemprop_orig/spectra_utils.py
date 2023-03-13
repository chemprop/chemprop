from typing import List
import csv

from tqdm import trange
import numpy as np


def normalize_spectra(spectra: List[List[float]], phase_features: List[List[float]] = None, phase_mask: List[List[float]] = None, batch_size: int = 50, excluded_sub_value: float = None, threshold: float = None) -> List[List[float]]:
    """
    Function takes in spectra and normalize them to sum values to 1. If provided with phase mask information, will remove excluded spectrum regions.

    :param spectra: Input spectra with shape (num_spectra, spectrum_length).
    :param phase_features: The collection phase of spectrum with shape (num_spectra, num_phases).
    :param phase_mask: A mask array showing where in each phase feature to include in predictions and training with shape (num_phases, spectrum_length)
    :param batch_size: The size of batches to carry out the normalization operation in.
    :param exlcuded_sub_value: Excluded values are replaced with this object, usually None or nan.
    :param threshold: Spectra values below threshold are replaced with threshold to remove negative or zero values.
    :return: List form array of spectra with shape (num_spectra, spectrum length) with exlcuded values converted to nan.
    """
    normalized_spectra = []
    phase_exclusion = phase_mask is not None and phase_features is not None
    if phase_exclusion:
        phase_mask = np.array(phase_mask)
    
    num_iters, iter_step = len(spectra), batch_size

    for i in trange(0, num_iters, iter_step):
        # prepare batch
        batch_spectra = spectra[i:i + iter_step]
        batch_mask = np.array([[x is not None for x in b] for b in batch_spectra])
        batch_spectra = np.array([[0 if x is None else x for x in b] for b in batch_spectra])
        if phase_exclusion:
            batch_phases = phase_features[i:i + iter_step]
            batch_phases = np.array(batch_phases)

        # exclude mask and apply threshold
        if threshold is not None:
            batch_spectra[batch_spectra < threshold] = threshold
        if phase_exclusion:
            batch_phase_mask = np.matmul(batch_phases, phase_mask).astype('bool')
            batch_mask = ~(~batch_mask + ~batch_phase_mask) # mask shows True only if both components true
        batch_spectra[~batch_mask] = 0
        
        # normalize to sum to 1
        sum_spectra = np.sum(batch_spectra, axis=1, keepdims=True)
        batch_spectra = batch_spectra / sum_spectra

        # Collect vectors and revert excluded values to None
        batch_spectra = batch_spectra.astype('object')
        batch_spectra[~batch_mask] = excluded_sub_value
        batch_spectra = batch_spectra.tolist()
        normalized_spectra.extend(batch_spectra)
    
    return normalized_spectra


def roundrobin_sid(spectra: np.ndarray, threshold: float = None) -> List[float]:
    """
    Takes a block of input spectra and makes a pairwise comparison between each of the input spectra for a given molecule,
    returning a list of the spectral informations divergences. To be used evaluating the variation between an ensemble of model spectrum predictions.

    :spectra: A 3D array containing each of the spectra to be compared. Shape of (num_spectra, spectrum_length, ensemble_size)
    :threshold: SID calculation requires positive values in each position, this value is used to replace any zero or negative values.
    :return: A list of average pairwise SID len (num_spectra)
    """
    ensemble_size=spectra.shape[2]
    spectrum_size=spectra.shape[1]

    ensemble_sids=[]

    for i in range(len(spectra)):
        spectrum = spectra[i]
        nan_mask=np.isnan(spectrum[:,0])
        if threshold is not None:
            spectrum[spectrum < threshold] = threshold
        spectrum[nan_mask,:]=1
        ensemble_head = np.zeros([spectrum_size,0])
        ensemble_tail = np.zeros([spectrum_size,0])
        for j in range(ensemble_size-1):
            ensemble_tail = np.concatenate((ensemble_tail,spectrum[:,j+1:]),axis=1)
            ensemble_head = np.concatenate((ensemble_head,spectrum[:,:-j-1]),axis=1)
        loss = ensemble_head * np.log(ensemble_head / ensemble_tail) + ensemble_tail * np.log(ensemble_tail / ensemble_head)
        loss[nan_mask,:]=0
        loss = np.sum(loss,axis=0)
        loss = np.mean(loss)
        ensemble_sids.append(loss)
    return ensemble_sids


def load_phase_mask(path: str) -> List[List[int]]:
    """
    Loads in a matrix used to mark sections of spectra as untrainable due to interference caused by particular phases.
    Ignore those spectra regions in training and prediciton.

    :param path: Path to a csv file containing the phase mask in shape (num_phases, spectrum_length) with 1s indicating inclusion and 0s indicating exclusion.
    :return: A list form array of the phase mask.
    """
    if path is None:
        return None

    data = []
    with open(path,'r') as rf:
        r=csv.reader(rf)
        next(r)
        for line in r:
            if any([x not in ['0','1'] for x in line[1:]]):
                raise ValueError('Phase mask must contain only 0s and 1s, with 0s indicating exclusion regions.')
            data_line = [int(x) for x in line[1:]]
            data.append(data_line)
    return data
