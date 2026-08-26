"""Nominal dynamics and OMPL adapters for every supported system."""

from propagators import (
    double_integrator,
    dubins_airplane,
    kinematic_car,
    pushing_object,
)
from propagators.double_integrator import DoubleIntegrator
from propagators.dubins_airplane import DubinsAirplane
from propagators.kinematic_car import KinematicCar
from propagators.propagator import System
from propagators.pushing_object import PushingObject


SYSTEMS = {
    "double_integrator": DoubleIntegrator,
    "dubins_airplane": DubinsAirplane,
    "kinematic_car": KinematicCar,
    "pushing_object": PushingObject,
}


def get_system(system_name: str) -> System:
    """Create a system from its canonical configuration name."""

    name = str(system_name).strip().lower()
    try:
        return SYSTEMS[name]()
    except KeyError as exc:
        supported = ", ".join(sorted(SYSTEMS))
        raise ValueError(
            f"Unknown system {system_name!r}; expected one of: {supported}"
        ) from exc


__all__ = [
    "DoubleIntegrator",
    "DubinsAirplane",
    "KinematicCar",
    "PushingObject",
    "System",
    "double_integrator",
    "dubins_airplane",
    "get_system",
    "kinematic_car",
    "pushing_object",
]
