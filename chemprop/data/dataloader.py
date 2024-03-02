import warnings

from torch.utils.data import DataLoader

from chemprop.data.collate import collate_batch, collate_multicomponent
from chemprop.data.datasets import MoleculeDataset, MulticomponentDataset, ReactionDataset
from chemprop.data.samplers import ClassBalanceSampler, SeededSampler


class MolGraphDataLoader(DataLoader):
    """A :class:`MolGraphDataLoader` is a :obj:`~torch.utils.data.DataLoader` for
    :class:`MolGraphDataset`\s

    Parameters
    ----------
    dataset : MoleculeDataset
        The dataset containing the molecules to load.
    batch_size : int, default=50
        the batch size to load
    num_workers : int, default=0
        the number of workers used to build batches.
    class_balance : bool, default=False
        Whether to perform class balancing (i.e., use an equal number of positive and negative
        molecules). Class balance is only available for single task classification datasets. Set
        shuffle to True in order to get a random subset of the larger class.
    seed : int, default=None
        the random seed to use for shuffling (only used when `shuffle` is `True`)
    shuffle : bool, default=False
        whether to shuffle the data during sampling
    """

    def __init__(
        self,
        dataset: MoleculeDataset | ReactionDataset | MulticomponentDataset,
        batch_size: int = 50,
        num_workers: int = 0,
        class_balance: bool = False,
        seed: int | None = None,
        shuffle: bool = True,
        **kwargs,
    ):
        if class_balance:
            sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
        elif shuffle and seed is not None:
            sampler = SeededSampler(len(dataset), seed)
        else:
            sampler = None

        if isinstance(dataset, MulticomponentDataset):
            collate_fn = collate_multicomponent
        else:
            collate_fn = collate_batch

        if len(dataset) % batch_size == 1:
            warnings.warn(
                f"Dropping last batch of size 1 to avoid issues with batch normalization \
(dataset size = {len(dataset)}, batch_size = {batch_size})"
            )
            drop_last = True
        else:
            drop_last = False

        super().__init__(
            dataset,
            batch_size,
            sampler is None and shuffle,
            sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
