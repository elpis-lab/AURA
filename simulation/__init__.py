"""Simulation and MuJoCo support package."""

from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parent
ASSET_DIR = SIMULATION_DIR / "assets"


def asset_path(*parts: str) -> str:
    return str(ASSET_DIR.joinpath(*parts))
