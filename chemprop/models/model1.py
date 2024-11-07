from __future__ import annotations

from typing import Iterable
import warnings

from lightning import pytorch as pl
import torch
from torch import Tensor, distributed, nn, optim

from torch_geometric.nn import GINConv, global_mean_pool 
from torch_geometric.datasets import TUDataset 
from torch_geometric.loader import DataLoader



from chemprop.data import BatchMolGraph, TrainingBatch
from chemprop.nn import Aggregation, LossFunction, MessagePassing, Predictor
from chemprop.nn.metrics import Metric
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched


class MPNN(pl.LightningModule):



    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> LossFunction:
        return self.predictor.criterion

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None, i: int = -1
    ) -> Tensor:
        """Calculate the :attr:`i`-th hidden representation"""
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.predictor(self.fingerprint(bmg, V_d, X_d))

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        self.log("train_loss", l, prog_bar=True)

        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.predictor.output_transform.train()

    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0):
        losses = self._evaluate_batch(batch)
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, batch_size=len(batch[0]))
        self.log(
            "val_loss",
            losses[0],
            batch_size=len(batch[0]),
            prog_bar=True,
            sync_dist=distributed.is_initialized(),
        )

    def test_step(self, batch: TrainingBatch, batch_idx: int = 0):
        losses = self._evaluate_batch(batch)
        metric2loss = {f"batch_averaged_test/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, batch_size=len(batch[0]))

    def _evaluate_batch(self, batch) -> list[Tensor]:
        bmg, V_d, X_d, targets, _, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        weights = torch.ones_like(targets)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        return [
            metric(preds, targets, mask, weights, lt_mask, gt_mask) for metric in self.metrics[:-1]
        ]

    def predict_step(self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Return the predictions of the input batch

        Parameters
        ----------
        batch : TrainingBatch
            the input batch

        Returns
        -------
        Tensor
            a tensor of varying shape depending on the task type:

            * regression/binary classification: ``n x (t * s)``, where ``n`` is the number of input
              molecules/reactions, ``t`` is the number of tasks, and ``s`` is the number of targets
              per task. The final dimension is flattened, so that the targets for each task are
              grouped. I.e., the first ``t`` elements are the first target for each task, the second
              ``t`` elements the second target, etc.

            * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
        """
        bmg, X_vd, X_d, *_ = batch

        return self(bmg, X_vd, X_d)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    @classmethod
    def load_submodules(cls, checkpoint_path, map_location=None, **kwargs):
        hparams = torch.load(checkpoint_path, map_location=map_location)["hyper_parameters"]

        kwargs |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in kwargs
        }
        return kwargs

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        kwargs = cls.load_submodules(checkpoint_path, map_location, **kwargs)
        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, strict, **kwargs
        )

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True) -> MPNN:
        d = torch.load(model_path, map_location=map_location)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {model_path}. ")

        for key in ["message_passing", "agg", "predictor"]:
            hparam_kwargs = hparams[key]
            hparam_cls = hparam_kwargs.pop("cls")
            hparams[key] = hparam_cls(**hparam_kwargs)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model



class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp))
            in_channels = hidden_channels
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None, i: int = -1
    ) -> Tensor:
        """Calculate the :attr:`i`-th hidden representation"""
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)
    

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.mlp(x)

    dataset = TUDataset(root='path/to/dataset', name='MUTAG')
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.9)], batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):], batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GIN(in_channels=dataset.num_node_features, hidden_channels=32, out_channels=dataset.num_classes, num_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = torch.nn.functional.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)

    def test(loader):
        model.eval()
        total_correct = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            total_correct += (pred == data.y).sum().item()
        return total_correct / len(loader.dataset)

    for epoch in range(1, 101):
        loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
