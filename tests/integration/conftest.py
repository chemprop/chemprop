import warnings

import pytest

from chemprop import models, nn
from chemprop.models import multi

warnings.filterwarnings("ignore", module=r"lightning.*", append=True)


@pytest.fixture(scope="session")
def mpnn(request):
    agg = nn.SumAggregation()
    ffn = nn.RegressionFFN()

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_mpnn(request):
    agg = nn.SumAggregation()
    ffn = nn.BinaryClassificationFFN()

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def mcmpnn(request):
    blocks, n_components, shared = request.param
    mcmp = nn.MulticomponentMessagePassing(blocks, n_components, shared=shared)
    agg = nn.SumAggregation()
    ffn = nn.RegressionFFN(input_dim=mcmp.output_dim,)

    return multi.MulticomponentMPNN(mcmp, agg, ffn, True)
