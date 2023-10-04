from torch.utils.data import DataLoader

from chemprop.v2.data.collate import collate_batch
from chemprop.v2.data.datasets import MolGraphDataset
from chemprop.v2.data.samplers import ClassBalanceSampler, SeededSampler


class MolGraphDataLoader(DataLoader):
    """A :class:`MolGraphDataLoader` is a :obj:`~torch.utils.data.DataLoader` for
    :class:`MolGraphDataset`s

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
        dataset: MolGraphDataset,
        batch_size: int = 50,
        num_workers: int = 0,
        class_balance: bool = False,
        seed: int | None = None,
        shuffle: bool = True,
    ):
        self.dset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        if self.class_balance:
            self.sampler = ClassBalanceSampler(self.dset.Y, seed, self.shuffle)
        elif self.shuffle and seed is not None:
            self.sampler = SeededSampler(len(self.dset), seed)
        else:
            self.sampler = None

        super().__init__(
            self.dset,
            batch_size,
            self.sampler is None and self.shuffle,
            self.sampler,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
