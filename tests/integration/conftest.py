import warnings

from lightning import pytorch as pl
import pytest

from chemprop import models, nn
from chemprop.models import multi

warnings.filterwarnings("ignore", module=r"lightning.*", append=True)


@pytest.fixture(scope="session")
def mpnn(request):
    message_passing, agg, *act = request.param
    ffn = nn.RegressionFFN(activation=act[0] if len(act) > 0 else "RELU")
    return models.MPNN(message_passing, agg, ffn, True)


@pytest.fixture(scope="session")
def mol_atom_bond_mpnn(request):
    pl.seed_everything(0)
    message_passing, agg = request.param
    mol_ffn = nn.RegressionFFN()
    atom_ffn = nn.RegressionFFN()
    bond_ffn = nn.RegressionFFN(input_dim=600)

    return models.MolAtomBondMPNN(
        message_passing, agg, mol_ffn, atom_ffn, bond_ffn, batch_norm=True
    )


@pytest.fixture(scope="session")
def regression_mpnn_mve(request):
    agg = nn.SumAggregation()
    ffn = nn.MveFFN()

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def regression_mpnn_evidential(request):
    agg = nn.SumAggregation()
    ffn = nn.EvidentialFFN()

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def regression_mpnn_quantile(request):
    agg = nn.SumAggregation()
    ffn = nn.QuantileFFN()

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_mpnn_dirichlet(request):
    agg = nn.SumAggregation()
    ffn = nn.BinaryDirichletFFN()

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_mpnn(request):
    agg = nn.SumAggregation()
    ffn = nn.BinaryClassificationFFN()

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_mpnn_multiclass(request):
    agg = nn.SumAggregation()
    ffn = nn.MulticlassClassificationFFN(n_classes=3)

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def classification_mpnn_multiclass_dirichlet(request):
    agg = nn.SumAggregation()
    ffn = nn.MulticlassDirichletFFN(n_classes=3)

    return models.MPNN(request.param, agg, ffn, True)


@pytest.fixture(scope="session")
def mcmpnn(request):
    blocks, n_components, shared = request.param
    mcmp = nn.MulticomponentMessagePassing(blocks, n_components, shared=shared)
    agg = nn.SumAggregation()
    ffn = nn.RegressionFFN(input_dim=mcmp.output_dim)

    return multi.MulticomponentMPNN(mcmp, agg, ffn, True)
