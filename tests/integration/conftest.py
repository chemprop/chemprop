import warnings

import pytest

from chemprop import models, nn
from chemprop.featurizers.molgraph import CondensedGraphOfReactionFeaturizer, SimpleMoleculeMolGraphFeaturizer

warnings.filterwarnings("ignore", module=r"lightning.*", append=True)


# @pytest.fixture(
#     params=[
#         SimpleMoleculeMolGraphFeaturizer(), CondensedGraphOfReactionFeaturizer()
#     ]
# )
# def featurizer(request):
#     return request.param


# @pytest.fixture(params=[nn.BondMessagePassing, nn.AtomMessagePassing])
# def mp_cls(request):
#     return request.param


# @pytest.fixture
# def mp(mp_cls, request):
#     d_v, d_e = request.shape

#     return mp_cls(d_v, d_e)


@pytest.fixture(scope="session")
def mpnn(request):
    agg = nn.SumAggregation()
    ffn = nn.RegressionFFN()

    return models.MPNN(request.param, agg, ffn, True)