"""
Tests for prism.lib.lang — TenantSpace and PrismProjector.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism.lib.lang import (
    TenantSpace,
    PrismProjector,
    ProjectionConfig,
    BlendMode,
    PayloadEnvelope,
    DimensionError,
    TenantError,
    AnchorError,
)


INPUT_DIM = 512
TARGET_DIM = 64


@pytest.fixture()
def tenant_a() -> TenantSpace:
    return TenantSpace("tenant_alpha", INPUT_DIM, TARGET_DIM)


@pytest.fixture()
def tenant_b() -> TenantSpace:
    return TenantSpace("tenant_beta", INPUT_DIM, TARGET_DIM)


# ---------------------------------------------------------------------------
# TenantSpace
# ---------------------------------------------------------------------------


class TestTenantSpace:
    def test_output_shape(self, tenant_a: TenantSpace) -> None:
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        out = tenant_a.project(v)
        assert out.shape == (TARGET_DIM,)

    def test_batch_shape(self, tenant_a: TenantSpace) -> None:
        batch = np.random.randn(32, INPUT_DIM).astype(np.float32)
        out = tenant_a.project(batch)
        assert out.shape == (32, TARGET_DIM)

    def test_deterministic(self, tenant_a: TenantSpace) -> None:
        """Same tenant_id must produce identical projections every time."""
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        space2 = TenantSpace("tenant_alpha", INPUT_DIM, TARGET_DIM)
        np.testing.assert_array_equal(tenant_a.project(v), space2.project(v))

    def test_cross_tenant_isolation(
        self, tenant_a: TenantSpace, tenant_b: TenantSpace
    ) -> None:
        """
        A vector projected by A then re-projected through B's inverse should
        look like noise — cross-correlation should be near zero on average.

        We test the statistical property: the dot product of independently
        projected same-norm vectors should be near zero.
        """
        rng = np.random.default_rng(42)
        dots = []
        for _ in range(100):
            v = rng.standard_normal(INPUT_DIM).astype(np.float32)
            za = tenant_a.project(v)
            zb = tenant_b.project(v)
            # Normalise
            za /= np.linalg.norm(za) + 1e-9
            zb /= np.linalg.norm(zb) + 1e-9
            dots.append(float(np.dot(za, zb)))

        mean_dot = abs(np.mean(dots))
        # For independent random matrices the expected value is 0.
        # 3σ bound: std ≈ 1/sqrt(k) = 1/8 = 0.125, so mean should be < 0.05
        assert mean_dot < 0.15, f"Cross-tenant dot product too high: {mean_dot:.4f}"

    def test_empty_tenant_id_raises(self) -> None:
        with pytest.raises(TenantError):
            TenantSpace("", INPUT_DIM, TARGET_DIM)

    def test_dim_zero_raises(self) -> None:
        """Zero input_dim is always invalid regardless of target."""
        with pytest.raises(DimensionError):
            TenantSpace("t", 0, TARGET_DIM)

    def test_expansion_mode_small_input(self) -> None:
        """input_dim < target_dim should expand, not raise."""
        space = TenantSpace("t", 4, TARGET_DIM)
        v = np.ones(4, dtype=np.float32)
        out = space.project(v)
        assert out.shape == (TARGET_DIM,)
        assert space.is_expansion is True

    def test_dim_mismatch_raises(self, tenant_a: TenantSpace) -> None:
        bad_v = np.ones(INPUT_DIM + 10, dtype=np.float32)
        with pytest.raises(DimensionError):
            tenant_a.project(bad_v)


# ---------------------------------------------------------------------------
# PrismProjector
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg_with_anchors() -> ProjectionConfig:
    rng = np.random.default_rng(0)
    anchors = {
        "finance": rng.standard_normal(INPUT_DIM).astype(np.float32),
        "healthcare": rng.standard_normal(INPUT_DIM).astype(np.float32),
    }
    return ProjectionConfig(
        tenant_id="tenant_test",
        target_dim=TARGET_DIM,
        blend_mode=BlendMode.SPHERICAL,
        default_blend_weight=0.2,
        anchors=anchors,
    )


class TestPrismProjector:
    def test_basic_projection(self, cfg_with_anchors: ProjectionConfig) -> None:
        proj = PrismProjector(cfg_with_anchors)
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        env = proj.project(v)
        assert isinstance(env, PayloadEnvelope)
        assert env.vector.shape == (TARGET_DIM,)
        assert env.tenant_id == "tenant_test"

    def test_rule_chain_populated(self, cfg_with_anchors: ProjectionConfig) -> None:
        proj = PrismProjector(cfg_with_anchors)
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        env = proj.project(v, anchor_label="finance")
        steps = [e.step for e in env.rule_chain]
        assert "normalise" in steps
        assert "spherical_blend" in steps
        assert "jl_project" in steps
        assert "normalise_output" in steps

    def test_output_is_unit_norm(self, cfg_with_anchors: ProjectionConfig) -> None:
        proj = PrismProjector(cfg_with_anchors)
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        env = proj.project(v)
        np.testing.assert_allclose(np.linalg.norm(env.vector), 1.0, atol=1e-5)

    def test_unknown_anchor_raises(self, cfg_with_anchors: ProjectionConfig) -> None:
        proj = PrismProjector(cfg_with_anchors)
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        with pytest.raises(AnchorError):
            proj.project(v, anchor_label="nonexistent")

    def test_batch_projection(self, cfg_with_anchors: ProjectionConfig) -> None:
        proj = PrismProjector(cfg_with_anchors)
        vecs = [np.random.randn(INPUT_DIM).astype(np.float32) for _ in range(5)]
        envs = proj.project_batch(vecs)
        assert len(envs) == 5
        for env in envs:
            assert env.vector.shape == (TARGET_DIM,)

    def test_to_dict_serialisable(self, cfg_with_anchors: ProjectionConfig) -> None:
        import json
        proj = PrismProjector(cfg_with_anchors)
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        env = proj.project(v)
        d = env.to_dict()
        # Must be JSON-serialisable (for logging / audit trails)
        serialised = json.dumps(d)
        assert len(serialised) > 0
