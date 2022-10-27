from chemprop.v2.utils.mixins import FactoryMixin

from .message_passing import *
from .readout import *

class ReadoutFactory(Readout, FactoryMixin):
    pass
