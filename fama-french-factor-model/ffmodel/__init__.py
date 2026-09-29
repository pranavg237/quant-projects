"""Fama-French factor models: data, time-series regressions, asset pricing tests, attribution."""
import warnings

# urllib3 2.x warns on import when Python is linked against LibreSSL (macOS system Python); harmless here.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from .asset_pricing import FamaMacBethResult, GRSResult, fama_macbeth, grs_test
from .assets import build_portfolio, download_returns, load_returns_csv, prices_to_returns
from .attribution import Attribution, attribute_returns
from .data import fetch_dataset, load_factors, load_portfolios, parse_french_csv
from .models import MODELS, FactorModel, get_model
from .regression import (
    RegressionResult,
    compare_models,
    compare_standard_errors,
    fit_factor_model,
    fit_many,
    holm_adjust,
    rolling_regression,
    stability_test,
    summarize,
)
from .snapshot import read_manifest, save_snapshot

__version__ = "0.1.0"
