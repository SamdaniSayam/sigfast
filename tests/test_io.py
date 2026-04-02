"""
tests/test_io.py
────────────────
Test suite for triples_sigfast.io — RootReader and SimReader.

Strategy: no real ROOT/FLUKA/MCNP/SERPENT files needed.
- RootReader: tested via uproot.writing to create real in-memory ROOT files.
- SimReader:  tested via temporary text files for FLUKA/MCNP/SERPENT backends,
              and format-detection tested via extension sniffing.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from triples_sigfast.io.sim_reader import SimReader, _detect_format


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_root_file(path: str, histograms: dict) -> None:
    """Create a real ROOT file with TH1F histograms using uproot.writing."""
    import uproot.writing as urw

    with urw.recreate(path) as f:
        for name, (counts, edges) in histograms.items():
            f[name] = (counts.astype(np.float64), edges.astype(np.float64))


def make_fluka_file(path: str, detectors: dict) -> str:
    lines = []
    for det_name, (energies, counts) in detectors.items():
        lines.append(f"# DETECTOR: {det_name}")
        for e, c in zip(energies, counts):
            lines.append(f"{e:.4f}  {c:.4f}")
        lines.append("#")
    content = "\n".join(lines)
    with open(path, "w") as f:
        f.write(content)
    return content


def make_mctal_file(path: str) -> None:
    content = """mctal
ntal 1
tally      4
f4:n  1
et
 0.0 0.5 1.0 2.0 5.0
vals
 1.23e-4 0.05 4.56e-4 0.03 7.89e-4 0.02 2.11e-3 0.01 5.00e-4 0.04
"""
    with open(path, "w") as f:
        f.write(content)


def make_serpent_file(path: str, n_bins: int = 3) -> None:
    rows = []
    for i in range(n_bins):
        e_low  = float(i)
        e_high = float(i + 1)
        row    = [e_low, e_high, 0.5, 1e-4, 0.05,
                  0.0,   0.0,    0.0, 0.0,  0.0,
                  float(i + 1) * 1e-3, 0.03]
        rows.append("  ".join(f"{v:.6E}" for v in row))
    block = "\n".join(rows)
    content = f"DET_NEUTRON_FLUX = [\n{block}\n];\n"
    with open(path, "w") as f:
        f.write(content)


# ── Format detection ──────────────────────────────────────────────────────────

class TestDetectFormat:

    def test_root_extension(self):
        assert _detect_format("sim.root") == "geant4"

    def test_flair_extension(self):
        assert _detect_format("sim.flair") == "fluka"

    def test_lis_extension(self):
        assert _detect_format("sim.lis") == "fluka"

    def test_mctal_extension(self):
        assert _detect_format("sim.mctal") == "mcnp"

    def test_det_extension(self):
        assert _detect_format("sim.det") == "serpent"

    def test_m_extension(self):
        assert _detect_format("sim.m") == "serpent"

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            _detect_format("sim.xyz")

    def test_case_insensitive(self):
        assert _detect_format("SIM.ROOT") == "geant4"


# ── RootReader ────────────────────────────────────────────────────────────────

class TestRootReader:

    @pytest.fixture
    def root_file(self, tmp_path):
        path = str(tmp_path / "test.root")
        rng  = np.random.default_rng(0)
        counts = rng.integers(100, 1000, size=100).astype(np.float64)
        edges  = np.linspace(0, 10, 101)
        make_root_file(path, {"neutron": (counts, edges),
                               "gamma":   (counts * 0.5, edges)})
        return path

    def test_repr(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        assert "RootReader" in repr(r)

    def test_keys_returns_list(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        assert isinstance(r.keys(), list)
        assert len(r.keys()) >= 2

    def test_histogram_keys(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        hkeys = r.histogram_keys()
        assert len(hkeys) >= 2

    def test_get_spectrum_exact_key(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        key = r.keys()[0]
        counts, energies = r.get_spectrum(key)
        assert counts.dtype == np.float64
        assert energies.dtype == np.float64
        assert len(counts) == len(energies)
        assert len(counts) == 100

    def test_get_spectrum_partial_key(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        counts, energies = r.get_spectrum("neutron")
        assert len(counts) == 100

    def test_get_spectrum_missing_key_raises(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        with pytest.raises(KeyError):
            r.get_spectrum("nonexistent_histogram_xyz")

    def test_get_all_spectra(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        all_s = r.get_all_spectra()
        assert len(all_s) >= 2
        for key, (counts, energies) in all_s.items():
            assert len(counts) == len(energies)

    def test_bin_centres_within_edges(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        counts, energies = r.get_spectrum("neutron")
        assert energies[0]  >= 0.0
        assert energies[-1] <= 10.0

    def test_export_csv(self, root_file, tmp_path):
        from triples_sigfast.io.root_reader import RootReader
        r      = RootReader(root_file)
        outcsv = str(tmp_path / "out.csv")
        r.export_csv(outcsv)
        assert os.path.exists(outcsv)
        import pandas as pd
        df = pd.read_csv(outcsv)
        assert len(df) == 100
        assert len(df.columns) >= 4

    def test_export_hdf5(self, root_file, tmp_path):
        pytest.importorskip("h5py")
        from triples_sigfast.io.root_reader import RootReader
        import h5py
        r     = RootReader(root_file)
        outh5 = str(tmp_path / "out.h5")
        r.export_hdf5(outh5)
        assert os.path.exists(outh5)
        with h5py.File(outh5, "r") as f:
            assert len(f.keys()) >= 2

    def test_context_manager(self, root_file):
        from triples_sigfast.io.root_reader import RootReader
        with RootReader(root_file) as r:
            counts, _ = r.get_spectrum("neutron")
        assert len(counts) == 100

    def test_summary_runs_without_error(self, root_file, capsys):
        from triples_sigfast.io.root_reader import RootReader
        r = RootReader(root_file)
        r.summary()
        captured = capsys.readouterr()
        assert "ROOT file" in captured.out


# ── SimReader — Geant4 ────────────────────────────────────────────────────────

class TestSimReaderGeant4:

    @pytest.fixture
    def root_file(self, tmp_path):
        path   = str(tmp_path / "sim.root")
        rng    = np.random.default_rng(1)
        counts = rng.integers(100, 500, size=50).astype(np.float64)
        edges  = np.linspace(0, 5, 51)
        make_root_file(path, {"gamma_spectrum": (counts, edges)})
        return path

    def test_format_detected(self, root_file):
        r = SimReader(root_file)
        assert r.format == "geant4"

    def test_repr(self, root_file):
        r = SimReader(root_file)
        assert "geant4" in repr(r)

    def test_get_spectrum(self, root_file):
        r = SimReader(root_file)
        counts, energies = r.get_spectrum()
        assert len(counts) == len(energies) == 50

    def test_keys(self, root_file):
        r = SimReader(root_file)
        assert len(r.keys()) >= 1


# ── SimReader — FLUKA ─────────────────────────────────────────────────────────

class TestSimReaderFLUKA:

    @pytest.fixture
    def fluka_file(self, tmp_path):
        path     = str(tmp_path / "sim.flair")
        energies = np.linspace(0.1, 10.0, 20)
        counts   = np.abs(np.random.default_rng(2).standard_normal(20)) * 100
        make_fluka_file(path, {"neutron_fluence": (energies, counts)})
        return path

    def test_format_detected(self, fluka_file):
        r = SimReader(fluka_file)
        assert r.format == "fluka"

    def test_get_spectrum_default(self, fluka_file):
        r = SimReader(fluka_file)
        counts, energies = r.get_spectrum()
        assert len(counts) > 0
        assert len(counts) == len(energies)

    def test_get_tally(self, fluka_file):
        r      = SimReader(fluka_file)
        tally  = r.get_tally("neutron_fluence")
        assert "values" in tally
        assert "bins"   in tally
        assert len(tally["values"]) > 0

    def test_keys(self, fluka_file):
        r = SimReader(fluka_file)
        assert "neutron_fluence" in r.keys()

    def test_summary(self, fluka_file, capsys):
        SimReader(fluka_file).summary()
        assert "FLUKA" in capsys.readouterr().out


# ── SimReader — MCNP ──────────────────────────────────────────────────────────

class TestSimReaderMCNP:

    @pytest.fixture
    def mctal_file(self, tmp_path):
        path = str(tmp_path / "sim.mctal")
        make_mctal_file(path)
        return path

    def test_format_detected(self, mctal_file):
        r = SimReader(mctal_file)
        assert r.format == "mcnp"

    def test_get_spectrum(self, mctal_file):
        r = SimReader(mctal_file)
        counts, energies = r.get_spectrum()
        assert len(counts) > 0

    def test_get_tally_by_name(self, mctal_file):
        r     = SimReader(mctal_file)
        tally = r.get_tally("tally_4")
        assert "values" in tally
        assert "errors" in tally

    def test_tally_errors_finite(self, mctal_file):
        r     = SimReader(mctal_file)
        tally = r.get_tally("tally_4")
        assert np.all(np.isfinite(tally["errors"]))

    def test_missing_tally_raises(self, mctal_file):
        r = SimReader(mctal_file)
        with pytest.raises(KeyError):
            r.get_tally("nonexistent_tally_999")

    def test_summary(self, mctal_file, capsys):
        SimReader(mctal_file).summary()
        assert "MCNP" in capsys.readouterr().out


# ── SimReader — SERPENT ───────────────────────────────────────────────────────

class TestSimReaderSerpent:

    @pytest.fixture
    def serpent_file(self, tmp_path):
        path = str(tmp_path / "sim.det")
        make_serpent_file(path, n_bins=5)
        return path

    def test_format_detected(self, serpent_file):
        r = SimReader(serpent_file)
        assert r.format == "serpent"

    def test_get_spectrum(self, serpent_file):
        r = SimReader(serpent_file)
        counts, energies = r.get_spectrum()
        assert len(counts) == 5
        assert len(energies) == 5

    def test_bin_centres_correct(self, serpent_file):
        r = SimReader(serpent_file)
        _, energies = r.get_spectrum()
        np.testing.assert_allclose(energies[0], 0.5, rtol=1e-6)
        np.testing.assert_allclose(energies[1], 1.5, rtol=1e-6)

    def test_get_tally(self, serpent_file):
        r     = SimReader(serpent_file)
        tally = r.get_tally("DET_NEUTRON_FLUX")
        assert "values" in tally
        assert len(tally["values"]) == 5

    def test_summary(self, serpent_file, capsys):
        SimReader(serpent_file).summary()
        assert "SERPENT" in capsys.readouterr().out

    def test_missing_key_raises(self, serpent_file):
        r = SimReader(serpent_file)
        with pytest.raises(KeyError):
            r.get_tally("nonexistent_detector_xyz")
