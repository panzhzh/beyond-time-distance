"""Forecast-shape dependence calibration with fixed marginal distributions."""
from .base import ChangePairs, DependenceConfig, ForecastInputs
from .calibration import CalibratedHeads, NativeConfig, fit_calibration
from .kernel import CalibrationBatch, FitConfig, GeometryRefinement
from .reference import TemporalReference, fit_temporal_reference
from .scoring import calibrated_knots, inverse_quantiles, native_change_scores, reconstructed_evidence

__version__ = '0.1.0'
__all__ = ['CalibrationBatch', 'CalibratedHeads', 'ChangePairs', 'DependenceConfig', 'FitConfig',
           'ForecastInputs', 'GeometryRefinement', 'NativeConfig', 'TemporalReference',
           'fit_calibration', 'fit_temporal_reference', 'calibrated_knots', 'inverse_quantiles',
           'native_change_scores', 'reconstructed_evidence']
