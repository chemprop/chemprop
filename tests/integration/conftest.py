import warnings

import pytest

from chemprop import models, nn

warnings.filterwarnings("ignore", module=r"lightning.*", append=True)


@pytest.fixture(scope="session")
def mpnn(request):
    agg = nn.SumAggregation()
    ffn = nn.RegressionFFN()

    return models.MPNN(request.param, agg, ffn, True)
