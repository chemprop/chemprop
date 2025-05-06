import logging
from os import PathLike

from lightning.pytorch import __version__
from lightning.pytorch.utilities.parsing import AttributeDict
import torch

from chemprop.nn.agg import AggregationRegistry
from chemprop.nn.message_passing import AtomMessagePassing, BondMessagePassing
from chemprop.nn.metrics import LossFunctionRegistry, MetricRegistry
from chemprop.nn.predictors import PredictorRegistry
from chemprop.nn.transforms import UnscaleTransform
from chemprop.utils import Factory

logger = logging.getLogger(__name__)


def convert_state_dict_v1_to_v2(model_v1_dict: dict) -> dict:
    """Converts v1 model dictionary to a v2 state dictionary"""

    state_dict_v2 = {}
    args_v1 = model_v1_dict["args"]

    state_dict_v1 = model_v1_dict["state_dict"]
    state_dict_v2["message_passing.W_i.weight"] = state_dict_v1["encoder.encoder.0.W_i.weight"]
    state_dict_v2["message_passing.W_h.weight"] = state_dict_v1["encoder.encoder.0.W_h.weight"]
    state_dict_v2["message_passing.W_o.weight"] = state_dict_v1["encoder.encoder.0.W_o.weight"]
    state_dict_v2["message_passing.W_o.bias"] = state_dict_v1["encoder.encoder.0.W_o.bias"]

    # v1.6 renamed ffn to readout
    if "readout.1.weight" in state_dict_v1:
        for i in range(args_v1.ffn_num_layers):
            suffix = 0 if i == 0 else 2
            state_dict_v2[f"predictor.ffn.{i}.{suffix}.weight"] = state_dict_v1[
                f"readout.{i * 3 + 1}.weight"
            ]
            state_dict_v2[f"predictor.ffn.{i}.{suffix}.bias"] = state_dict_v1[
                f"readout.{i * 3 + 1}.bias"
            ]
    else:
        for i in range(args_v1.ffn_num_layers):
            suffix = 0 if i == 0 else 2
            state_dict_v2[f"predictor.ffn.{i}.{suffix}.weight"] = state_dict_v1[
                f"ffn.{i * 3 + 1}.weight"
            ]
            state_dict_v2[f"predictor.ffn.{i}.{suffix}.bias"] = state_dict_v1[
                f"ffn.{i * 3 + 1}.bias"
            ]

    if args_v1.dataset_type == "regression":
        state_dict_v2["predictor.output_transform.mean"] = torch.tensor(
            model_v1_dict["data_scaler"]["means"], dtype=torch.float32
        ).unsqueeze(0)
        state_dict_v2["predictor.output_transform.scale"] = torch.tensor(
            model_v1_dict["data_scaler"]["stds"], dtype=torch.float32
        ).unsqueeze(0)

    # target_weights was added in #183
    if getattr(args_v1, "target_weights", None) is not None:
        task_weights = torch.tensor(args_v1.target_weights).unsqueeze(0)
    else:
        task_weights = torch.ones(args_v1.num_tasks).unsqueeze(0)

    state_dict_v2["predictor.criterion.task_weights"] = task_weights

    return state_dict_v2


def convert_hyper_parameters_v1_to_v2(model_v1_dict: dict) -> dict:
    """Converts v1 model dictionary to v2 hyper_parameters dictionary"""
    hyper_parameters_v2 = {}
    renamed_metrics = {
        "auc": "roc",
        "prc-auc": "prc",
        "cross_entropy": "ce",
        "binary_cross_entropy": "bce",
        "mcc": "binary-mcc",
        "recall": "recall is not in v2",
        "precision": "precision is not in v2",
        "balanced_accuracy": "balanced_accuracy is not in v2",
    }

    args_v1 = model_v1_dict["args"]
    hyper_parameters_v2["batch_norm"] = False
    hyper_parameters_v2["metrics"] = [
        Factory.build(MetricRegistry[renamed_metrics.get(args_v1.metric, args_v1.metric)])
    ]
    hyper_parameters_v2["warmup_epochs"] = args_v1.warmup_epochs
    hyper_parameters_v2["init_lr"] = args_v1.init_lr
    hyper_parameters_v2["max_lr"] = args_v1.max_lr
    hyper_parameters_v2["final_lr"] = args_v1.final_lr

    # convert the message passing block
    W_i_shape = model_v1_dict["state_dict"]["encoder.encoder.0.W_i.weight"].shape
    W_h_shape = model_v1_dict["state_dict"]["encoder.encoder.0.W_h.weight"].shape
    W_o_shape = model_v1_dict["state_dict"]["encoder.encoder.0.W_o.weight"].shape

    d_h = W_i_shape[0]
    d_v = W_o_shape[1] - d_h
    d_e = W_h_shape[1] - d_h if args_v1.atom_messages else W_i_shape[1] - d_v

    hyper_parameters_v2["message_passing"] = AttributeDict(
        {
            "activation": args_v1.activation,
            "bias": args_v1.bias,
            "cls": BondMessagePassing if not args_v1.atom_messages else AtomMessagePassing,
            "d_e": d_e,  # the feature dimension of the edges
            "d_h": args_v1.hidden_size,  # dimension of the hidden layer
            "d_v": d_v,  # the feature dimension of the vertices
            "d_vd": args_v1.atom_descriptors_size,
            "depth": args_v1.depth,
            "dropout": args_v1.dropout,
            "undirected": args_v1.undirected,
        }
    )

    # convert the aggregation block
    hyper_parameters_v2["agg"] = {
        "dim": 0,  # in v1, the aggregation is always done on the atom features
        "cls": AggregationRegistry[args_v1.aggregation],
    }
    if args_v1.aggregation == "norm":
        hyper_parameters_v2["agg"]["norm"] = args_v1.aggregation_norm

    # convert the predictor block
    fgs = args_v1.features_generator or []
    d_xd = sum((200 if "rdkit" in fg else 0) + (2048 if "morgan" in fg else 0) for fg in fgs)

    if getattr(args_v1, "target_weights", None) is not None:
        task_weights = torch.tensor(args_v1.target_weights).unsqueeze(0)
    else:
        task_weights = torch.ones(args_v1.num_tasks).unsqueeze(0)

    # loss_function was added in #238
    loss_fn_defaults = {
        "classification": "bce",
        "regression": "mse",
        "multiclass": "ce",
        "specitra": "sid",
    }
    T_loss_fn = LossFunctionRegistry[
        getattr(args_v1, "loss_function", loss_fn_defaults[args_v1.dataset_type])
    ]

    hyper_parameters_v2["predictor"] = AttributeDict(
        {
            "activation": args_v1.activation,
            "cls": PredictorRegistry[args_v1.dataset_type],
            "criterion": Factory.build(T_loss_fn, task_weights=task_weights),
            "task_weights": None,
            "dropout": args_v1.dropout,
            "hidden_dim": args_v1.ffn_hidden_size,
            "input_dim": args_v1.hidden_size + args_v1.atom_descriptors_size + d_xd,
            "n_layers": args_v1.ffn_num_layers - 1,
            "n_tasks": args_v1.num_tasks,
        }
    )

    if args_v1.dataset_type == "regression":
        hyper_parameters_v2["predictor"]["output_transform"] = UnscaleTransform(
            model_v1_dict["data_scaler"]["means"], model_v1_dict["data_scaler"]["stds"]
        )

    return hyper_parameters_v2


def convert_model_dict_v1_to_v2(model_v1_dict: dict) -> dict:
    """Converts a v1 model dictionary from a loaded .pt file to a v2 model dictionary"""

    model_v2_dict = {}

    model_v2_dict["epoch"] = None
    model_v2_dict["global_step"] = None
    model_v2_dict["pytorch-lightning_version"] = __version__
    model_v2_dict["state_dict"] = convert_state_dict_v1_to_v2(model_v1_dict)
    model_v2_dict["loops"] = None
    model_v2_dict["callbacks"] = None
    model_v2_dict["optimizer_states"] = None
    model_v2_dict["lr_schedulers"] = None
    model_v2_dict["hparams_name"] = "kwargs"
    model_v2_dict["hyper_parameters"] = convert_hyper_parameters_v1_to_v2(model_v1_dict)

    return model_v2_dict


def convert_model_file_v1_to_v2(model_v1_file: PathLike, model_v2_file: PathLike) -> None:
    """Converts a v1 model .pt file to a v2 model .pt file"""

    model_v1_dict = torch.load(model_v1_file, map_location=torch.device("cpu"), weights_only=False)
    model_v2_dict = convert_model_dict_v1_to_v2(model_v1_dict)
    logger.warning(
        "Remember to use the same featurizers which were used when training the model. The default "
        "v1 atom featurizer is `chemprop.featurizers.atom.MultiHotAtomFeaturizer.v1()` and can be "
        "specified from the command line with `--multi-hot-atom-featurizer-mode v1`."
    )
    torch.save(model_v2_dict, model_v2_file)
