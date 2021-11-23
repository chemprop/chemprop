from argparse import Namespace
from typing import List, Optional, Union, Tuple
import warnings

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights
from chemprop.models.mpn import MPN


class _MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(
        self,
        dataset_type: str,
        num_tasks: int,
        multiclass_num_classes: int,
        atom_messages: bool = False,
        hidden_size: int = 300,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = False,
        device: Union[str, torch.device] = "cpu",
        aggregation: str = "mean",
        aggregation_norm: int = 100,
        checkpoint_frzn: Optional[str] = None,
        freeze_first_only: bool = False,
        frzn_ffn_layers: int = 0,
        features_only: bool = False,
        features_size: Optional[int] = None,
        number_of_molecules: int = 1,
        use_input_features: int = False,
        atom_descriptors: Optional[str] = None,
        atom_descriptors_size: int = 0,
        activation: str = "ReLU",
        ffn_num_layers: int = 2,
        ffn_hidden_size: Optional[int] = None,
        spectra_activation: str = "exp",
        **kwargs,
    ):
        super().__init__()

        self.classification = dataset_type == "classification"
        self.multiclass = dataset_type == "multiclass"

        self.output_size = num_tasks

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.output_size *= multiclass_num_classes
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(
            atom_messages,
            hidden_size,
            bias,
            depth,
            dropout,
            undirected,
            device,
            aggregation,
            aggregation_norm,
            checkpoint_frzn,
            freeze_first_only,
        )
        self.create_ffn(
            dataset_type,
            multiclass_num_classes,
            features_only,
            features_size,
            hidden_size,
            number_of_molecules,
            use_input_features,
            atom_descriptors,
            atom_descriptors_size,
            dropout,
            activation,
            ffn_num_layers,
            ffn_hidden_size,
            spectra_activation,
            checkpoint_frzn,
            frzn_ffn_layers,
        )

        initialize_weights(self)

    def create_encoder(
        self,
        atom_messages: bool = False,
        hidden_size: int = 300,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = False,
        device: Union[str, torch.device] = "cpu",
        aggregation: str = "mean",
        aggregation_norm: int = 100,
        checkpoint_frzn: Optional[str] = None,
        freeze_first_only: bool = False,
    ):
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param checkpoint_frzn:
        :param freezer_first_only:
        """
        self.encoder = MPN(
            Namespace(
                atom_messages=atom_messages,
                hidden_size=hidden_size,
                bias=bias,
                depth=depth,
                dropout=dropout,
                layers_per_message=1,
                undirected=undirected,
                device=device,
                aggregation=aggregation,
                aggregation_norm=aggregation_norm,
            )
        )

        if checkpoint_frzn is not None:
            if freeze_first_only:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def create_ffn(
        self,
        dataset_type: str,
        multiclass_num_classes: int,
        features_only: bool = False,
        features_size: Optional[int] = None,
        hidden_size: int = 300,
        number_of_molecules: int = 1,
        use_input_features: int = False,
        atom_descriptors: Optional[str] = None,
        atom_descriptors_size: int = 0,
        dropout: float = 0.0,
        activation: str = "ReLU",
        ffn_num_layers: int = 2,
        ffn_hidden_size: Optional[int] = None,
        spectra_activation: str = "exp",
        checkpoint_frzn: Optional[str] = None,
        frzn_ffn_layers: int = 0,
    ):
        """Creates the feed-forward layers for the model.

        :param dataset_type: the type of dataset this model will be used for
        :type dataset_type: str
        :param multiclass_num_classes: the number of classes to predict for, if using this model
            for multiclass prediction
        :type multiclass_num_classes: int
        :param features_only: use only the additional features in the FFN, no MPN features
        :type features_only: bool
        :param features_size: the dimension of the additional molecule features
        :type features_size: int
        :param hidden_size: the size of the hidden layers in the MPN
        :type hidden_size: int
        :param number_of_molecules: the number of molecules in each input to the model
        :type number_of_molecules: int
        :param use_input_features: whether to use additional input features
        :type use_input_features: int
        :param atom_descriptors: the custom additional atom descriptors
        :type atom_descriptors: str
        :param atom_descriptors_size: the size of the additional atom descriptors, if they are used
        :type atom_descriptors_size: int
        :param dropout: the dropout probability for model training
        :type dropout: float
        :param activation: the activation function to use
        :type activation: str
        :param ffn_num_layers: the number of layers in the FFN
        :type ffn_num_layers: int
        :param ffn_hidden_size: the size of each hidden layer in the FFN
        :type ffn_hidden_size: int
        :param spectra_activation: the activation function to use for for spectral datasets
        :type spectra_activation: str
        :param checkpoint_frzn: the filepath to a model checkpoint file to write to for freezing
            weights. NOTE: will overwrite previously existing files
        :type checkpoint_frzn: str
        :param frzn_ffn_layers: the number of FFN layers that should be frozen
        :type frzn_ffn_layers: int
        """
        if self.multiclass:
            self.num_classes = multiclass_num_classes

        if features_only:
            first_linear_dim = features_size
        else:
            first_linear_dim = hidden_size * number_of_molecules
            if use_input_features:
                first_linear_dim += features_size

        if atom_descriptors == "descriptor":
            first_linear_dim += atom_descriptors_size

        dropout = nn.Dropout(dropout)
        activation = get_activation_function(activation)

        # Create FFN layers
        if ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, self.output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, ffn_hidden_size)]
            for _ in range(ffn_num_layers - 2):
                ffn.extend(
                    [
                        activation,
                        dropout,
                        nn.Linear(ffn_hidden_size, ffn_hidden_size),
                    ]
                )
            ffn.extend(
                [
                    activation,
                    dropout,
                    nn.Linear(ffn_hidden_size, self.output_size),
                ]
            )

        # If spectra model, also include spectra activation
        if dataset_type == "spectra":
            if spectra_activation == "softplus":
                spectra_activation = nn.Softplus()
            else:  # default exponential activation which must be made into a custom nn module

                class nn_exp(torch.nn.Module):
                    def __init__(self):
                        super(nn_exp, self).__init__()

                    def forward(self, x):
                        return torch.exp(x)

                spectra_activation = nn_exp()
            ffn.append(spectra_activation)

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

        if checkpoint_frzn is not None and frzn_ffn_layers > 0:
            for param in list(self.ffn.parameters())[0:2*frzn_ffn_layers]:
                param.requires_grad = False

    def fingerprint(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        fingerprint_type="MPN",
    ) -> torch.FloatTensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == "MPN":
            return self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_features_batch,
            )
        elif fingerprint_type == "last_FFN":
            return self.ffn[:-1](
                self.encoder(
                    batch,
                    features_batch,
                    atom_descriptors_batch,
                    atom_features_batch,
                    bond_features_batch,
                )
            )
        else:
            raise ValueError(f"Unsupported fingerprint type {fingerprint_type}.")

    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
    ) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions
        """

        output = self.ffn(
            self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_features_batch,
            )
        )

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape(
                (output.size(0), -1, self.num_classes)
            )  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(
                    output
                )  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


class MoleculeModel(_MoleculeModel):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: Optional[TrainArgs] = None, *pargs, **kwargs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        if args is not None:
            warnings.warn(
                "Initializing a MoleculeModel using a TrainArgs object will be deprecated in "
                "a future version of Chemprop!",
                DeprecationWarning,
            )
            super().__init__(**vars(args))
        else:
            super().__init__(*pargs, **kwargs)
