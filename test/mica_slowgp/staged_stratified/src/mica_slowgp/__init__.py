"""Isolated two-band MICA-style Gaussian-transfer/slow-GP prototype."""

from .core import (
    COMPONENTS,
    MODES,
    PARAMETER_NAMES,
    SLOW_PARAMETER_NAMES,
    LikelihoodState,
    PairData,
    Prediction,
    build_covariance,
    component_covariance,
    condition,
    factorize,
    gaussian_ou_covariance,
    log_likelihood,
    matern32_covariance,
    validate_parameters,
)

__all__ = [
    "COMPONENTS", "MODES", "PARAMETER_NAMES", "SLOW_PARAMETER_NAMES",
    "LikelihoodState", "PairData", "Prediction", "build_covariance",
    "component_covariance", "condition", "factorize", "gaussian_ou_covariance",
    "log_likelihood", "matern32_covariance", "validate_parameters",
]
