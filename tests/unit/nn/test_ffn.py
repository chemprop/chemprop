import pytest
import torch

from chemprop.nn.ffn import MLP, ConstrainerFFN


class TestMLPBuild:
    """Tests for MLP.build() with per-layer hidden dimensions."""

    def test_legacy_single_int(self):
        mlp = MLP.build(input_dim=10, output_dim=2, hidden_dim=300, n_layers=2)
        x = torch.randn(4, 10)
        y = mlp(x)
        assert y.shape == (4, 2)
        assert mlp.input_dim == 10
        assert mlp.output_dim == 2

    def test_per_layer_funnel(self):
        mlp = MLP.build(input_dim=10, output_dim=2, hidden_dim=[600, 300, 150], n_layers=3)
        x = torch.randn(4, 10)
        y = mlp(x)
        assert y.shape == (4, 2)

    def test_per_layer_hourglass(self):
        mlp = MLP.build(input_dim=10, output_dim=2, hidden_dim=[512, 64, 512], n_layers=3)
        x = torch.randn(4, 10)
        y = mlp(x)
        assert y.shape == (4, 2)

    def test_per_layer_single_layer(self):
        mlp = MLP.build(input_dim=10, output_dim=1, hidden_dim=[256], n_layers=1)
        x = torch.randn(2, 10)
        y = mlp(x)
        assert y.shape == (2, 1)

    def test_per_layer_multi_output(self):
        mlp = MLP.build(input_dim=20, output_dim=5, hidden_dim=[128, 64])
        x = torch.randn(3, 20)
        y = mlp(x)
        assert y.shape == (3, 5)

    @pytest.mark.parametrize(
        "hidden_dims, n_layers", [([32, 64, 128], 3), ([512], 1), ([256, 128], 2)]
    )
    def test_per_layer_shapes(self, hidden_dims, n_layers):
        mlp = MLP.build(input_dim=16, output_dim=3, hidden_dim=hidden_dims, n_layers=n_layers)
        x = torch.randn(2, 16)
        y = mlp(x)
        assert y.shape == (2, 3)
        assert mlp.input_dim == 16
        assert mlp.output_dim == 3

    def test_per_layer_different_from_legacy(self):
        """Verify per-layer dims produce different param count than uniform expansion."""
        mlp_uniform = MLP.build(input_dim=10, output_dim=1, hidden_dim=300, n_layers=3)
        mlp_funnel = MLP.build(input_dim=10, output_dim=1, hidden_dim=[600, 300, 150], n_layers=3)
        uniform_params = sum(p.numel() for p in mlp_uniform.parameters())
        funnel_params = sum(p.numel() for p in mlp_funnel.parameters())
        assert uniform_params != funnel_params

    def test_legacy_expansion_equivalence(self):
        """A single int with n_layers=2 should match explicit [d, d]."""
        mlp_int = MLP.build(input_dim=10, output_dim=1, hidden_dim=128, n_layers=2)
        mlp_list = MLP.build(input_dim=10, output_dim=1, hidden_dim=[128, 128], n_layers=2)
        x = torch.randn(4, 10)
        with torch.no_grad():
            for p_int, p_list in zip(mlp_int.parameters(), mlp_list.parameters()):
                p_list.copy_(p_int)
        y_int = mlp_int(x)
        y_list = mlp_list(x)
        assert torch.allclose(y_int, y_list)


class TestConstrainerFFNPerLayer:
    """Tests for ConstrainerFFN with per-layer hidden dimensions."""

    def test_constrainer_per_layer(self):
        constrainer = ConstrainerFFN(n_constraints=2, fp_dim=300, hidden_dim=[128, 64], n_layers=2)
        fp = torch.randn(5, 300)
        preds = torch.randn(5, 2)
        batch = torch.tensor([0, 0, 1, 1, 1])
        constraints = torch.randn(2, 2)
        result = constrainer(fp, preds, batch, constraints)
        assert result.shape == (5, 2)

    def test_constrainer_per_layer_satisfies_constraint(self):
        batch = torch.tensor([0, 1, 1, 3, 3, 3])
        rows_per_group = torch.bincount(batch)
        b = len(batch)
        t = 3
        m = batch.max().item() + 1

        fp = torch.randn(b, 100)
        preds = torch.randn(b, t)
        constraints = torch.randn(m, t)
        constraints[2] = 0  # molecule 2 has no atoms in batch

        constrainer = ConstrainerFFN(
            n_constraints=t, fp_dim=100, hidden_dim=[64, 32], n_layers=2, dropout=0.0
        )
        with torch.no_grad():
            constrained_preds = constrainer(fp, preds, batch, constraints)
        constrained_preds = torch.split(constrained_preds, rows_per_group.tolist(), dim=0)
        constrained_preds = torch.stack([torch.sum(p, dim=0) for p in constrained_preds])
        assert torch.allclose(constrained_preds, constraints, rtol=1e-4, atol=1e-4)


class TestPredictorPerLayer:
    """Tests for _FFNPredictorBase subclasses with per-layer hidden dimensions."""

    @pytest.mark.parametrize("hidden_dims", [300, [256, 128], [512, 256, 128]])
    def test_regression_ffn_per_layer(self, hidden_dims):
        from chemprop.nn.predictors import RegressionFFN

        pred = RegressionFFN(n_tasks=2, input_dim=300, hidden_dim=hidden_dims)
        z = torch.randn(4, 300)
        out = pred(z)
        assert out.shape == (4, 2)
        assert pred.input_dim == 300

    @pytest.mark.parametrize("hidden_dims", [300, [256, 128]])
    def test_binary_classification_ffn_per_layer(self, hidden_dims):
        from chemprop.nn.predictors import BinaryClassificationFFN

        pred = BinaryClassificationFFN(n_tasks=1, input_dim=200, hidden_dim=hidden_dims)
        z = torch.randn(3, 200)
        out = pred(z)
        assert out.shape == (3, 1)
        assert (out >= 0).all() and (out <= 1).all()

    @pytest.mark.parametrize("hidden_dims", [[256, 128]])
    def test_multiclass_classification_ffn_per_layer(self, hidden_dims):
        from chemprop.nn.predictors import MulticlassClassificationFFN

        pred = MulticlassClassificationFFN(
            n_classes=3, n_tasks=2, input_dim=150, hidden_dim=hidden_dims
        )
        z = torch.randn(4, 150)
        out = pred(z)
        assert out.shape == (4, 2, 3)
        assert torch.allclose(out.sum(-1), torch.ones(4, 2), atol=1e-5)

    @pytest.mark.parametrize("hidden_dims", [300, [128, 64]])
    def test_spectral_ffn_per_layer(self, hidden_dims):
        from chemprop.nn.predictors import SpectralFFN

        pred = SpectralFFN(n_tasks=1, input_dim=200, hidden_dim=hidden_dims)
        z = torch.randn(2, 200)
        out = pred(z)
        assert out.shape == (2, 1)
        assert torch.allclose(out.sum(-1), torch.ones(2), atol=1e-5)
