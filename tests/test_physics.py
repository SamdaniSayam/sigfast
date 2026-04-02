# ============================================================
#  triples-sigfast — Physics Features Test Suite
# ============================================================

import numpy as np
import pytest

from triples_sigfast import (
    attenuation,
    attenuation_series,
    find_peaks,
    flux_to_dose,
    savitzky_golay,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def noisy_spectrum():
    np.random.seed(42)
    x = np.linspace(0, 10, 500)
    signal = np.exp(-((x - 2.5) ** 2) / 0.1) + np.exp(-((x - 7.0) ** 2) / 0.1)
    noise = np.random.randn(500) * 0.05
    return signal + noise


@pytest.fixture
def clean_peak():
    x = np.linspace(0, 10, 200)
    return np.exp(-((x - 5.0) ** 2) / 0.2)


# ── savitzky_golay ───────────────────────────────────────────


class TestSavitzkyGolay:
    def test_output_length(self, noisy_spectrum):
        result = savitzky_golay(noisy_spectrum, window=11, polyorder=3)
        assert len(result) == len(noisy_spectrum)

    def test_reduces_noise(self, noisy_spectrum):
        result = savitzky_golay(noisy_spectrum, window=11, polyorder=3)
        assert result.std() < noisy_spectrum.std()

    def test_preserves_peak_position(self, clean_peak):
        result = savitzky_golay(clean_peak, window=11, polyorder=3)
        assert np.argmax(result) == np.argmax(clean_peak)

    def test_invalid_even_window(self, noisy_spectrum):
        with pytest.raises(ValueError):
            savitzky_golay(noisy_spectrum, window=10)

    def test_invalid_polyorder_too_large(self, noisy_spectrum):
        with pytest.raises(ValueError):
            savitzky_golay(noisy_spectrum, window=5, polyorder=5)

    def test_invalid_data_too_short(self):
        with pytest.raises(ValueError):
            savitzky_golay([1.0, 2.0, 3.0], window=11)

    def test_accepts_list_input(self, noisy_spectrum):
        result = savitzky_golay(noisy_spectrum.tolist(), window=11, polyorder=3)
        assert len(result) == len(noisy_spectrum)

    def test_flat_signal_unchanged(self):
        flat = np.ones(100)
        result = savitzky_golay(flat, window=11, polyorder=3)
        assert np.allclose(result, flat, atol=1e-10)

    def test_accepts_pandas_series(self):
        import pandas as pd

        s = pd.Series(np.random.randn(100))
        result = savitzky_golay(s, window=11, polyorder=3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)


# ── find_peaks ───────────────────────────────────────────────


class TestFindPeaks:
    def test_finds_known_peaks(self, clean_peak):
        peaks = find_peaks(clean_peak, min_height=0.5)
        assert len(peaks) == 1

    def test_finds_two_peaks(self, noisy_spectrum):
        smoothed = savitzky_golay(noisy_spectrum, window=21, polyorder=3)
        peaks = find_peaks(smoothed, min_height=0.5, min_distance=50)
        assert len(peaks) == 2

    def test_min_height_filters_small_peaks(self):
        data = np.array([0.1, 0.5, 0.1, 0.2, 0.1, 0.9, 0.1])
        peaks = find_peaks(data, min_height=0.4)
        assert len(peaks) == 2

    def test_min_height_too_high_no_peaks(self):
        data = np.array([0.1, 0.5, 0.1, 0.9, 0.1])
        peaks = find_peaks(data, min_height=2.0)
        assert len(peaks) == 0

    def test_min_distance_enforced(self):
        data = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        peaks_close = find_peaks(data, min_height=0.5, min_distance=1)
        peaks_far = find_peaks(data, min_height=0.5, min_distance=5)
        assert len(peaks_close) >= len(peaks_far)

    def test_invalid_min_distance(self, clean_peak):
        with pytest.raises(ValueError):
            find_peaks(clean_peak, min_distance=0)

    def test_returns_numpy_array(self, clean_peak):
        peaks = find_peaks(clean_peak, min_height=0.5)
        assert isinstance(peaks, np.ndarray)

    def test_flat_signal_no_peaks(self):
        flat = np.ones(100)
        peaks = find_peaks(flat, min_height=0.5)
        assert len(peaks) == 0


# ── flux_to_dose ─────────────────────────────────────────────


class TestFluxToDose:
    def test_neutron_returns_positive(self):
        dose = flux_to_dose(flux=1e6, energy_mev=1.0, particle="neutron")
        assert dose > 0

    def test_gamma_returns_positive(self):
        dose = flux_to_dose(flux=1e6, energy_mev=1.0, particle="gamma")
        assert dose > 0

    def test_higher_flux_higher_dose(self):
        low = flux_to_dose(flux=1e4, energy_mev=1.0, particle="neutron")
        high = flux_to_dose(flux=1e6, energy_mev=1.0, particle="neutron")
        assert high > low

    def test_cf252_average_energy(self):
        dose = flux_to_dose(flux=1e6, energy_mev=2.35, particle="neutron")
        assert dose > 0

    def test_array_flux_input(self):
        flux_array = np.array([1e4, 1e5, 1e6, 1e7])
        doses = flux_to_dose(flux=flux_array, energy_mev=1.0, particle="neutron")
        assert len(doses) == 4
        assert np.all(np.diff(doses) > 0)

    def test_invalid_particle(self):
        with pytest.raises(ValueError):
            flux_to_dose(flux=1e6, energy_mev=1.0, particle="electron")

    def test_invalid_energy_zero(self):
        with pytest.raises(ValueError):
            flux_to_dose(flux=1e6, energy_mev=0.0, particle="neutron")

    def test_case_insensitive_particle(self):
        d1 = flux_to_dose(flux=1e6, energy_mev=1.0, particle="Neutron")
        d2 = flux_to_dose(flux=1e6, energy_mev=1.0, particle="neutron")
        assert np.isclose(d1, d2)

    def test_units_are_reasonable(self):
        dose = flux_to_dose(flux=1e6, energy_mev=1.0, particle="neutron")
        assert 0.01 < dose < 10000


# ── attenuation ──────────────────────────────────────────────


class TestAttenuation:
    def test_zero_thickness_full_transmission(self):
        T = attenuation(thickness_cm=0, material="lead")
        assert np.isclose(T, 1.0)

    def test_transmission_decreases_with_thickness(self):
        T1 = attenuation(thickness_cm=1, material="lead")
        T10 = attenuation(thickness_cm=10, material="lead")
        assert T10 < T1

    def test_transmission_between_zero_and_one(self):
        T = attenuation(thickness_cm=5, material="lead")
        assert 0.0 < T < 1.0

    def test_lead_blocks_more_than_polyethylene(self):
        T_lead = attenuation(thickness_cm=5, material="lead")
        T_poly = attenuation(thickness_cm=5, material="polyethylene")
        assert T_lead < T_poly

    def test_composite_shield(self):
        T = attenuation(5, "lead") * attenuation(10, "polyethylene")
        assert 0.0 < T < 1.0

    def test_all_builtin_materials(self):
        materials = [
            "lead",
            "polyethylene",
            "concrete",
            "water",
            "iron",
            "bismuth",
            "tungsten",
            "borated_poly",
            "polysulfone",
        ]
        for mat in materials:
            T = attenuation(thickness_cm=5, material=mat)
            assert 0.0 < T < 1.0

    def test_custom_material(self):
        T = attenuation(thickness_cm=5, mu_rho=0.08, density=2.5)
        assert 0.0 < T < 1.0

    def test_custom_material_missing_density(self):
        with pytest.raises(ValueError):
            attenuation(thickness_cm=5, mu_rho=0.08)

    def test_unknown_material(self):
        with pytest.raises(ValueError):
            attenuation(thickness_cm=5, material="unobtainium")

    def test_negative_thickness(self):
        with pytest.raises(ValueError):
            attenuation(thickness_cm=-1, material="lead")

    def test_attenuation_series_output_length(self):
        thicknesses = np.linspace(0, 20, 100)
        result = attenuation_series(thicknesses, material="lead")
        assert len(result) == 100

    def test_attenuation_series_monotone_decreasing(self):
        thicknesses = np.linspace(0.1, 20, 50)
        result = attenuation_series(thicknesses, material="lead")
        assert np.all(np.diff(result) < 0)


# ── Public API ───────────────────────────────────────────────


class TestPublicAPI:
    def test_savitzky_golay_importable(self):
        from triples_sigfast import savitzky_golay

        assert callable(savitzky_golay)

    def test_find_peaks_importable(self):
        from triples_sigfast import find_peaks

        assert callable(find_peaks)

    def test_flux_to_dose_importable(self):
        from triples_sigfast import flux_to_dose

        assert callable(flux_to_dose)

    def test_attenuation_importable(self):
        from triples_sigfast import attenuation

        assert callable(attenuation)

    def test_attenuation_series_importable(self):
        from triples_sigfast import attenuation_series

        assert callable(attenuation_series)
