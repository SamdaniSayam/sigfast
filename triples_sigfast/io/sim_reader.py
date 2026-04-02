"""
triples_sigfast.io.sim_reader
─────────────────────────────
Universal simulation file reader.

Detects the output format of Geant4, FLUKA, MCNP, and SERPENT automatically
from the file extension and exposes a single consistent API regardless of
the underlying format.

Supported formats
-----------------
.root           -> Geant4 (via RootReader / uproot)
.flair / .lis   -> FLUKA  (text-based, parsed natively)
.mctal          -> MCNP   (text-based, parsed natively)
.det / .m       -> SERPENT (MATLAB-style text, parsed natively)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# ── Format detection ──────────────────────────────────────────────────────────

_EXT_MAP: dict[str, str] = {
    ".root":  "geant4",
    ".flair": "fluka",
    ".lis":   "fluka",
    ".mctal": "mcnp",
    ".det":   "serpent",
    ".m":     "serpent",
}


def _detect_format(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext not in _EXT_MAP:
        raise ValueError(
            f"Unrecognised file extension '{ext}'. "
            f"Supported: {list(_EXT_MAP.keys())}"
        )
    return _EXT_MAP[ext]


# ── SimReader ─────────────────────────────────────────────────────────────────

class SimReader:
    """
    Universal reader for simulation output files.

    Automatically detects the simulation code from the file extension and
    delegates to the appropriate backend. All backends expose the same API.

    Parameters
    ----------
    filepath : str
        Path to the simulation output file. Extension determines the backend:
        .root -> Geant4, .flair/.lis -> FLUKA, .mctal -> MCNP, .det/.m -> SERPENT.

    Examples
    --------
    >>> reader = SimReader("output.root")    # Geant4
    >>> reader = SimReader("output.flair")   # FLUKA
    >>> reader = SimReader("output.mctal")   # MCNP
    >>> reader = SimReader("output.det")     # SERPENT

    >>> spectrum = reader.get_spectrum()
    >>> tally    = reader.get_tally("neutron_flux")
    >>> reader.summary()
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.format = _detect_format(filepath)
        self._backend = self._load_backend(filepath, self.format)

    def _load_backend(self, filepath: str, fmt: str):
        if fmt == "geant4":
            from triples_sigfast.io.root_reader import RootReader
            return RootReader(filepath)
        if fmt == "fluka":
            return _FlukaBackend(filepath)
        if fmt == "mcnp":
            return _MCNPBackend(filepath)
        if fmt == "serpent":
            return _SerpentBackend(filepath)
        raise ValueError(f"No backend for format: {fmt}")  # pragma: no cover

    # ── Unified API ───────────────────────────────────────────────────────

    def get_spectrum(
        self,
        key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1-D energy spectrum as (counts, bin_centres).

        Parameters
        ----------
        key : str, optional
            Histogram / tally name. If None, returns the first available
            spectrum in the file.

        Returns
        -------
        counts : np.ndarray
        bin_centres : np.ndarray
        """
        return self._backend.get_spectrum(key)

    def get_tally(self, name: str) -> dict:
        """
        Retrieve a named tally result.

        Returns a dict with keys: 'values', 'errors', 'bins', 'name'.
        The exact content depends on the simulation code.
        """
        return self._backend.get_tally(name)

    def summary(self) -> None:
        """Print a human-readable summary of available data in the file."""
        self._backend.summary()

    def keys(self) -> list[str]:
        """Return all available keys / tally names in the file."""
        return self._backend.keys()

    def __repr__(self) -> str:
        return f"SimReader('{self.filepath}', format='{self.format}')"


# ── FLUKA backend ─────────────────────────────────────────────────────────────

class _FlukaBackend:
    """
    Minimal FLUKA text output parser.

    Parses USRBIN / energy deposition output from .flair or .lis files.
    Full FLUKA reader (Month 3) will replace this with native binaries.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._parse()

    def _parse(self) -> None:
        with open(self.filepath) as f:
            lines = f.readlines()

        current_key: str | None = None
        energies: list[float] = []
        counts: list[float] = []

        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                if current_key and energies:
                    self._data[current_key] = (
                        np.array(counts,   dtype=np.float64),
                        np.array(energies, dtype=np.float64),
                    )
                    energies, counts = [], []
                if line.startswith("# DETECTOR:"):
                    current_key = line.split(":")[-1].strip()
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    energies.append(float(parts[0]))
                    counts.append(float(parts[1]))
                except ValueError:
                    continue

        if current_key and energies:
            self._data[current_key] = (
                np.array(counts,   dtype=np.float64),
                np.array(energies, dtype=np.float64),
            )

    def get_spectrum(self, key=None):
        if not self._data:
            raise RuntimeError(f"No data parsed from {self.filepath}")
        if key is None:
            key = next(iter(self._data))
        if key not in self._data:
            matches = [k for k in self._data if key in k]
            if not matches:
                raise KeyError(f"Key '{key}' not found. Available: {list(self._data)}")
            key = matches[0]
        return self._data[key]

    def get_tally(self, name: str) -> dict:
        counts, energies = self.get_spectrum(name)
        return {
            "name":   name,
            "values": counts,
            "errors": np.zeros_like(counts),
            "bins":   energies,
        }

    def summary(self) -> None:
        print(f"\nFLUKA file: {self.filepath}")
        print(f"  Detectors found: {len(self._data)}")
        for key, (counts, energies) in self._data.items():
            print(f"  {key:<30} {len(counts)} bins, integral={counts.sum():.4f}")
        print()

    def keys(self) -> list[str]:
        return list(self._data.keys())


# ── MCNP backend ──────────────────────────────────────────────────────────────

class _MCNPBackend:
    """
    MCTAL file parser for MCNP6 tally output.

    Parses the MCTAL ASCII format: tally headers, energy bins, and
    mean +/- relative-error pairs. Handles multi-line et and vals blocks.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._tallies: dict[str, dict] = {}
        self._parse()

    def _parse(self) -> None:
        with open(self.filepath) as f:
            content = f.read()

        tally_blocks = re.split(r"(?m)^tally\s+", content)[1:]

        for block in tally_blocks:
            lines = block.strip().splitlines()
            if not lines:
                continue

            header    = lines[0].split()
            tally_num = header[0] if header else "unknown"
            key       = f"tally_{tally_num}"

            energies: list[float] = []
            values:   list[float] = []
            errors:   list[float] = []
            mode: str | None = None  # 'energy' or 'vals'

            for line in lines[1:]:
                stripped = line.strip()
                if not stripped:
                    continue
                low = stripped.lower()

                if low.startswith("et"):
                    mode = "energy"
                    parts = stripped.split()[1:]
                    try:
                        energies.extend(float(p) for p in parts)
                    except ValueError:
                        pass
                    continue

                if low.startswith("vals"):
                    mode = "vals"
                    continue

                if mode == "energy":
                    try:
                        energies.extend(float(p) for p in stripped.split())
                    except ValueError:
                        mode = None

                elif mode == "vals":
                    parts = stripped.split()
                    try:
                        pairs = [float(p) for p in parts]
                        for i in range(0, len(pairs) - 1, 2):
                            values.append(pairs[i])
                            errors.append(pairs[i + 1])
                    except ValueError:
                        mode = None

            if values:
                n = min(len(energies), len(values))
                self._tallies[key] = {
                    "name":   key,
                    "values": np.array(values[:n],   dtype=np.float64),
                    "errors": np.array(errors[:n],   dtype=np.float64),
                    "bins":   np.array(energies[:n], dtype=np.float64),
                }

    def get_spectrum(self, key=None):
        if not self._tallies:
            raise RuntimeError(f"No tallies parsed from {self.filepath}")
        if key is None:
            key = next(iter(self._tallies))
        if key not in self._tallies:
            matches = [k for k in self._tallies if key in k]
            if not matches:
                raise KeyError(f"Key '{key}' not found. Available: {list(self._tallies)}")
            key = matches[0]
        t = self._tallies[key]
        return t["values"], t["bins"]

    def get_tally(self, name: str) -> dict:
        if name not in self._tallies:
            matches = [k for k in self._tallies if name in k]
            if not matches:
                raise KeyError(
                    f"Tally '{name}' not found. Available: {list(self._tallies)}"
                )
            name = matches[0]
        return self._tallies[name]

    def summary(self) -> None:
        print(f"\nMCNP MCTAL file: {self.filepath}")
        print(f"  Tallies found: {len(self._tallies)}")
        for key, t in self._tallies.items():
            print(f"  {key:<20} {len(t['values'])} energy bins")
        print()

    def keys(self) -> list[str]:
        return list(self._tallies.keys())


# ── SERPENT backend ───────────────────────────────────────────────────────────

class _SerpentBackend:
    """
    SERPENT detector output parser (.det / .m files).

    Parses MATLAB-style variable assignments produced by SERPENT2.
    Extracts detector arrays: energy bins, flux values, and errors.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._detectors: dict[str, dict] = {}
        self._parse()

    def _parse(self) -> None:
        with open(self.filepath) as f:
            content = f.read()

        pattern = re.compile(
            r"(\w+)\s*=\s*\[(.*?)\]\s*;",
            re.DOTALL,
        )

        for match in pattern.finditer(content):
            name   = match.group(1)
            values = re.findall(
                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(2)
            )
            if not values:
                continue

            arr = np.array([float(v) for v in values], dtype=np.float64)

            # SERPENT detector format: 12 columns per energy bin
            # Cols: E_low, E_high, lethargy, flux, rel_err, ...
            if len(arr) % 12 == 0 and len(arr) >= 12:
                arr = arr.reshape(-1, 12)
                self._detectors[name] = {
                    "name":   name,
                    "values": arr[:, 10],
                    "errors": arr[:, 11],
                    "bins":   0.5 * (arr[:, 0] + arr[:, 1]),
                    "raw":    arr,
                }
            else:
                self._detectors[name] = {
                    "name":   name,
                    "values": arr,
                    "errors": np.zeros_like(arr),
                    "bins":   np.arange(len(arr), dtype=np.float64),
                    "raw":    arr,
                }

    def get_spectrum(self, key=None):
        if not self._detectors:
            raise RuntimeError(f"No detectors parsed from {self.filepath}")
        if key is None:
            key = next(iter(self._detectors))
        if key not in self._detectors:
            matches = [k for k in self._detectors if key.upper() in k.upper()]
            if not matches:
                raise KeyError(
                    f"Key '{key}' not found. Available: {list(self._detectors)}"
                )
            key = matches[0]
        d = self._detectors[key]
        return d["values"], d["bins"]

    def get_tally(self, name: str) -> dict:
        if name not in self._detectors:
            matches = [k for k in self._detectors if name.upper() in k.upper()]
            if not matches:
                raise KeyError(
                    f"Detector '{name}' not found. Available: {list(self._detectors)}"
                )
            name = matches[0]
        return self._detectors[name]

    def summary(self) -> None:
        print(f"\nSERPENT file: {self.filepath}")
        print(f"  Detectors found: {len(self._detectors)}")
        for key, d in self._detectors.items():
            print(f"  {key:<30} {len(d['values'])} bins")
        print()

    def keys(self) -> list[str]:
        return list(self._detectors.keys())
