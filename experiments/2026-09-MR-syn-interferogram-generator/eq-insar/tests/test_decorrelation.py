"""Tests for generate_decorrelation_noise."""

import numpy as np
import pytest

from eq_insar.insar.noise import generate_decorrelation_noise


class TestDecorrelationNoise:

    def test_output_keys_and_shapes(self):
        res = generate_decorrelation_noise((64, 80), seed=0)
        for key in ["phase_noise_rad", "noise_m", "coherence",
                    "vegetation_weight", "snow_weight", "rupture_weight"]:
            assert res[key].shape == (64, 80)

    def test_phase_bounded(self):
        res = generate_decorrelation_noise((64, 64), background_coherence=0.0, seed=1)
        assert np.all(np.abs(res["phase_noise_rad"]) <= np.pi)

    def test_zero_coherence_is_uniform(self):
        res = generate_decorrelation_noise((256, 256), n_looks=1,
                                           background_coherence=0.0, seed=2)
        assert abs(res["phase_noise_rad"].std() - np.pi / np.sqrt(3)) < 0.03

    def test_full_coherence_is_noise_free(self):
        res = generate_decorrelation_noise((32, 32), background_coherence=1.0, seed=3)
        assert np.allclose(res["phase_noise_rad"], 0.0, atol=1e-12)

    def test_more_looks_less_noise(self):
        s1 = generate_decorrelation_noise((128, 128), n_looks=1, seed=4)["phase_noise_rad"].std()
        s16 = generate_decorrelation_noise((128, 128), n_looks=16, seed=4)["phase_noise_rad"].std()
        assert s16 < s1

    def test_converges_to_cramer_rao(self):
        g, L = 0.8, 32
        res = generate_decorrelation_noise((256, 256), n_looks=L,
                                           background_coherence=g, seed=5)
        crb = np.sqrt(1 - g**2) / (g * np.sqrt(2 * L))
        assert abs(res["phase_noise_rad"].std() - crb) / crb < 0.1

    def test_vegetation_fraction(self):
        res = generate_decorrelation_noise((128, 128), vegetation_fraction=0.3, seed=6)
        assert abs((res["vegetation_weight"] > 0.5).mean() - 0.3) < 0.02

    def test_longer_baseline_lower_coherence(self):
        kw = dict(shape=(64, 64), vegetation_fraction=0.5, seed=7)
        short = generate_decorrelation_noise(temporal_baseline_days=6, **kw)
        long = generate_decorrelation_noise(temporal_baseline_days=96, **kw)
        assert long["coherence"].mean() < short["coherence"].mean()

    def test_rupture_decorrelates_trace(self):
        res = generate_decorrelation_noise((101, 101), pixel_size_km=0.5,
                                           rupture_length_km=20, rupture_strike_deg=90,
                                           rupture_coherence=0.0, seed=8)
        assert res["coherence"][50, 50] < 0.01          # on trace
        assert res["coherence"][5, 50] > 0.85           # far from trace

    def test_user_mask(self):
        mask = np.zeros((32, 32)); mask[:, :16] = 1.0
        res = generate_decorrelation_noise((32, 32), snow_mask=mask,
                                           snow_coherence=0.0, seed=9)
        assert np.all(res["coherence"][:, :16] == 0.0)
        assert np.allclose(res["coherence"][:, 16:], 0.9)

    def test_reproducible(self):
        a = generate_decorrelation_noise((32, 32), vegetation_fraction=0.2, seed=10)
        b = generate_decorrelation_noise((32, 32), vegetation_fraction=0.2, seed=10)
        assert np.array_equal(a["phase_noise_rad"], b["phase_noise_rad"])

    def test_noise_m_conversion(self):
        res = generate_decorrelation_noise((32, 32), wavelength_m=0.2, seed=11)
        assert np.allclose(res["noise_m"], -res["phase_noise_rad"] * 0.2 / (4 * np.pi))

    @pytest.mark.parametrize("kw", [dict(n_looks=0), dict(background_coherence=1.5),
                                    dict(vegetation_fraction=-0.1),
                                    dict(snow_mask=np.ones((5, 5)))])
    def test_invalid_inputs(self, kw):
        with pytest.raises(ValueError):
            generate_decorrelation_noise((32, 32), **kw)


# ---------------------------------------------------------------------------
# Generator integration
# ---------------------------------------------------------------------------

from eq_insar import (
    generate_synthetic_insar,
    generate_timeseries,
    generate_training_batch,
    batch_to_arrays,
)
from eq_insar.core.moment_tensor import fault_dimensions_from_magnitude


class TestSingleIntegration:

    def test_default_has_no_decorrelation(self):
        res = generate_synthetic_insar(Mw=6.0, grid_size=32, seed=0)
        assert "coherence" not in res
        assert res["metadata"]["add_decorrelation"] is False

    def test_adds_coherence_and_changes_phase(self):
        kw = dict(Mw=6.0, grid_size=64, seed=0)
        base = generate_synthetic_insar(**kw)
        dec = generate_synthetic_insar(
            add_decorrelation=True,
            decorrelation_params=dict(vegetation_fraction=0.4), **kw)
        assert dec["coherence"].shape == (64, 64)
        assert set(dec["decorrelation_weights"]) == {"vegetation", "snow", "rupture"}
        assert not np.allclose(base["phase_noisy"], dec["phase_noisy"])
        assert np.all(np.abs(dec["phase_noisy"]) <= np.pi)

    def test_independent_of_add_noise(self):
        res = generate_synthetic_insar(
            Mw=6.0, grid_size=32, add_noise=False, add_decorrelation=True,
            decorrelation_params=dict(background_coherence=0.3), seed=1)
        assert not np.allclose(res["phase_noisy"], res["phase_wrapped"])

    def test_rupture_follows_source(self):
        res = generate_synthetic_insar(
            Mw=6.5, strike_deg=60, xcen_km=5, ycen_km=-5, grid_size=101,
            add_decorrelation=True,
            decorrelation_params=dict(rupture_length_km="auto"), seed=2)
        p = res["metadata"]["decorrelation_params"]
        assert p["rupture_strike_deg"] == 60
        assert p["rupture_center_km"] == (5, -5)
        assert np.isclose(p["rupture_length_km"],
                          fault_dimensions_from_magnitude(6.5)[0])
        # Lowest coherence lies near the epicentre
        X, Y, c = res["X_km"], res["Y_km"], res["coherence"]
        i = np.unravel_index(np.argmin(c), c.shape)
        assert np.hypot(X[i] - 5, Y[i] + 5) < 15

    def test_explicit_rupture_overrides(self):
        res = generate_synthetic_insar(
            Mw=6.0, strike_deg=60, grid_size=32, add_decorrelation=True,
            decorrelation_params=dict(rupture_length_km=10,
                                      rupture_strike_deg=10), seed=3)
        assert res["metadata"]["decorrelation_params"]["rupture_strike_deg"] == 10

    def test_bad_auto_string(self):
        with pytest.raises(ValueError):
            generate_synthetic_insar(
                Mw=6.0, grid_size=32, add_decorrelation=True,
                decorrelation_params=dict(rupture_length_km="big"))

    def test_reproducible(self):
        kw = dict(Mw=6.0, grid_size=32, add_decorrelation=True,
                  decorrelation_params=dict(vegetation_fraction=0.3), seed=4)
        assert np.array_equal(generate_synthetic_insar(**kw)["phase_noisy"],
                              generate_synthetic_insar(**kw)["phase_noisy"])


class TestTimeseriesIntegration:

    def _run(self, **kw):
        return generate_timeseries(
            Mw=6.5, grid_size=65, n_pre=2, n_event=1, n_post=2,
            add_decorrelation=True, seed=5, **kw)

    def test_coherence_shape(self):
        res = self._run(decorrelation_params=dict(vegetation_fraction=0.3))
        assert res["coherence"].shape == res["timeseries"].shape

    def test_rupture_only_in_event_frames(self):
        res = self._run(decorrelation_params=dict(rupture_length_km=20,
                                                  rupture_coherence=0.0))
        c = res["coherence"][:, 32, 32]
        assert c[2] < 0.05
        assert np.allclose(c[[0, 1, 3, 4]], 0.9)

    def test_vegetation_static_across_frames(self):
        res = self._run(decorrelation_params=dict(vegetation_fraction=0.3,
                                                  rupture_length_km=0))
        c = res["coherence"]
        assert all(np.allclose(c[0], c[t]) for t in range(1, len(c)))

    def test_snow_varies_across_frames(self):
        res = self._run(decorrelation_params=dict(snow_fraction=0.2))
        c = res["coherence"]
        assert not np.allclose(c[0], c[1])

    def test_noise_redrawn_each_frame(self):
        res = self._run(decorrelation_params=dict(background_coherence=0.2))
        ts = res["timeseries"]
        assert not np.allclose(ts[0], ts[1])

    def test_displacement_output(self):
        res = self._run(output_type="displacement",
                        decorrelation_params=dict(background_coherence=0.5))
        assert np.isfinite(res["timeseries"]).all()


class TestBatchIntegration:

    def test_batch_coherence_stacked(self):
        batch = generate_training_batch(
            3, grid_size=32, n_pre=1, n_event=1, n_post=1,
            add_decorrelation=True,
            decorrelation_params=dict(vegetation_fraction=0.2), seed=6,
            verbose=False)
        arr = batch_to_arrays(batch)
        assert arr["coherence"].shape == arr["X"].shape

    def test_batch_without_decorrelation(self):
        batch = generate_training_batch(2, grid_size=32, seed=6, verbose=False)
        assert "coherence" not in batch_to_arrays(batch)
