from typing import List, Optional
from warnings import warn

import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        return checkpoint_paths

    return None


class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    phase_features_path: str = None
    """Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    atom_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    bond_features_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""
    no_cache_mol: bool = False
    """
    Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    """
    empty_cache: bool = False
    """
    Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.
    """

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional molecule-level features.
        """
        return not self.no_features_scaling

    @features_scaling.setter
    def features_scaling(self, features_scaling: bool) -> None:
        self.no_features_scaling = not features_scaling

    @property
    def atom_features_size(self) -> int:
        """The size of the atom features."""
        return self._atom_features_size

    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size

    @property
    def atom_descriptors_size(self) -> int:
        """The size of the atom descriptors."""
        return self._atom_descriptors_size

    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size

    @property
    def bond_features_size(self) -> int:
        """The size of the atom features."""
        return self._bond_features_size

    @bond_features_size.setter
    def bond_features_size(self, bond_features_size: int) -> None:
        self._bond_features_size = bond_features_size

    def configure(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Validate features
        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator and self.features_scaling:
            raise ValueError('When using rdkit_2d_normalized features, --no_features_scaling must be specified.')

        # Validate atom descriptors
        if (self.atom_descriptors is None) != (self.atom_descriptors_path is None):
            raise ValueError('If atom_descriptors is specified, then an atom_descriptors_path must be provided '
                             'and vice versa.')

        if self.atom_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        # Validate bond descriptors
        if self.bond_features_path is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Bond descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        set_cache_mol(not self.no_cache_mol)

        if self.empty_cache:
            empty_cache()


class PredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    preds_path: str
    """Path to CSV file where predictions will be saved."""
    drop_extra_columns: bool = False
    """Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns."""
    ensemble_variance: bool = False
    """Deprecated. Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path."""
    individual_ensemble_predictions: bool = False
    """Whether to return the predictions made by each of the individual models rather than the average of the ensemble"""
    # Uncertainty arguments
    uncertainty_method: Literal[
        # todo: might need to rename to regression-mve
        'mve',
        'ensemble',
        'evidential_epistemic',
        'evidential_aleatoric',
        'evidential_total',
        'classification',
        'dropout',
        'spectra_roundrobin',
    ] = None
    """The method of calculating uncertainty."""
    calibration_method: Literal['zscaling', 'tscaling', 'zelikman_interval', 'mve_weighting', 'platt', 'isotonic'] = None
    """Methods used for calibrating the uncertainty calculated with uncertainty method."""
    evaluation_methods: List[str] = None
    """The methods used for evaluating the uncertainty performance if the test data provided includes targets.
    Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric."""
    evaluation_scores_path: str = None
    """Location to save the results of uncertainty evaluations."""
    uncertainty_dropout_p: float = 0.1
    """The probability to use for Monte Carlo dropout uncertainty estimation."""
    dropout_sampling_size: int = 10
    """The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training."""
    calibration_interval_percentile: float = 95
    """Sets the percentile used in the calibration methods. Must be in the range (1,100)."""
    regression_calibrator_metric: Literal['stdev', 'interval'] = None
    """Regression calibrators can output either a stdev or an inverval. """
    calibration_path: str = None
    """Path to data file to be used for uncertainty calibration."""
    calibration_features_path: str = None
    """Path to features data to be used with the uncertainty calibration dataset."""
    calibration_phase_features_path: str = None
    """ """
    calibration_atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    calibration_bond_features_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()

        if self.regression_calibrator_metric is None:
            if self.calibration_method == 'zelikman_interval':
                self.regression_calibrator_metric = 'interval'
            else:
                self.regression_calibrator_metric = 'stdev'

        if self.uncertainty_method == 'dropout' and version.parse(torch.__version__) < version.parse('1.9.0'):
            raise ValueError('Dropout uncertainty is only supported for pytorch versions >= 1.9.0')

        self.smiles_columns = chemprop.data.utils.preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')

        if self.ensemble_variance == True:
            if self.uncertainty_method in ['ensemble', None]:
                warn(
                    'The `--ensemble_variance` argument is deprecated and should \
                        be replaced with `--uncertainty_method ensemble`.',
                    DeprecationWarning,
                )
                self.uncertainty_method = 'ensemble'
            else:
                raise ValueError(
                    f'Only one uncertainty method can be used at a time. \
                        The arguement `--ensemble_variance` was provided along \
                        with the uncertainty method {self.uncertainty_method}. The `--ensemble_variance` \
                        argument is deprecated and should be replaced with `--uncertainty_method ensemble`.'
                )

        if self.calibration_interval_percentile <= 1 or self.calibration_interval_percentile >= 100:
            raise ValueError('The calibration interval must be a percentile value in the range (1,100).')

        if self.uncertainty_dropout_p < 0 or self.uncertainty_dropout_p > 1:
            raise ValueError('The dropout probability must be in the range (0,1).')

        if self.dropout_sampling_size <= 1:
            raise ValueError('The argument `--dropout_sampling_size` must be an integer greater than 1.')

        # Validate that features provided for the prediction test set are also provided for the calibration set
        for (features_argument, base_features_path, cal_features_path) in [
            ('`--features_path`', self.features_path, self.calibration_features_path),
            ('`--phase_features_path`', self.phase_features_path, self.calibration_phase_features_path),
            ('`--atom_descriptors_path`', self.atom_descriptors_path, self.calibration_atom_descriptors_path),
            ('`--bond_features_path`', self.bond_features_path, self.calibration_bond_features_path)
        ]:
            if base_features_path is not None and self.calibration_path is not None and cal_features_path is None:
                raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the calibration dataset.')
