from enum import auto

from chemprop.utils.utils import AutoName


class Aggregation(AutoName):
    MEAN = auto()
    NORM = auto()
    SUM = auto()
