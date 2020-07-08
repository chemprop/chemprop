import torch.nn as nn
import torch.nn.functional as F
import torch
from chemprop.distill.factory import RegisterDistill
from chemprop.nn_utils import get_activation_function

@RegisterDistill("base_distill")
class BaseDistill(nn.Module):
    def __init__(self, args):
        super(BaseDistill, self).__init__()

    def forward(self, x):
        return x, {}

    def compute_loss(self, output_dict, target_features_batch):
        return torch.zeros_like(target_features_batch)

    def additional_losses_to_log(self):
        return {}

@RegisterDistill("mse_distill")
class MseDistill(BaseDistill):
    def __init__(self, args):
        super(MseDistill, self).__init__(args)
        self.ffn = get_auxiliary_ffn(get_encoded_dim(args), args.target_features_size, args)
        self.mse = nn.MSELoss(reduction = 'none')
        self.args = args

    def forward(self, x, **kwargs):
        return x, {'student_z': self.ffn(x)}

    def compute_loss(self, output_dict, target_features_batch):
        return self.args.distill_lambda * self.mse(output_dict['student_z'], target_features_batch)

@RegisterDistill("pred_as_hidden_mse_distill")
class PredAsHiddenMseDistill(MseDistill):
    def forward(self, x, **kwargs):
        z = self.ffn(x)
        return z, {'student_z': z}


@RegisterDistill("prediction_distill")
class PredictionDistill(BaseDistill):
    def __init__(self, args):
        super(PredictionDistill, self).__init__(args)
        self.ffn = get_auxiliary_ffn(args.target_features_size, args.output_size, args)
        self.args = args

    def distill_loss_func(self, preds, targets):
        if self.args.dataset_type in ['classification', 'multiclass']:
            return F.kl_div(preds, torch.sigmoid(targets), reduction='none')
        else:
            return F.mse_loss(preds, targets, reduction='none')

    def compute_loss(self, output_dict, target_features_batch):
        teacher_y = self.ffn(target_features_batch)
        student_y = output_dict['logits']
        teacher_loss = output_dict['compute_loss_fn'](teacher_y)
        self.teacher_loss = teacher_loss.item()
        return teacher_loss + self.args.distill_lambda * self.distill_loss_func(teacher_y.detach(), student_y)

    def additional_losses_to_log(self):
        return {"teacher_loss": self.teacher_loss}


def get_encoded_dim(args):
    if args.features_only:
        first_linear_dim = args.features_size
    else:
        first_linear_dim = args.hidden_size
        if args.use_input_features:
            first_linear_dim += args.features_size

    return first_linear_dim


def get_auxiliary_ffn(first_linear_dim, output_dim, args):
    """
    Creates an FFN where input is x and output is z

    :param args: Arguments.
    """
    dropout = nn.Dropout(args.dropout)
    activation = get_activation_function(args.activation)

    # Create auxiliary FFN layers
    if args.ffn_num_layers == 1:
        ffn = [
            dropout,
            nn.Linear(first_linear_dim, output_dim)
        ]
    else:
        ffn = [
            dropout,
            nn.Linear(first_linear_dim, args.ffn_hidden_size)
        ]
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(args.ffn_hidden_size, output_dim),
        ])

    # Create FFN model
    return nn.Sequential(*ffn)
