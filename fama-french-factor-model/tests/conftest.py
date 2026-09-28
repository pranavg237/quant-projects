import numpy as np
import pandas as pd
import pytest

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def make_factors(T=600, seed=0, start="1970-01-31"):
    rng = np.random.default_rng(seed)
    means = np.array([0.006, 0.002, 0.003, 0.003, 0.002, 0.006])
    vols = np.array([0.045, 0.030, 0.030, 0.020, 0.020, 0.040])
    index = pd.date_range(start, periods=T, freq="ME", name="date")
    factors = pd.DataFrame(rng.normal(means, vols, size=(T, 6)), index=index, columns=FACTOR_NAMES)
    factors["RF"] = 0.003
    return factors


@pytest.fixture
def factors():
    return make_factors()


@pytest.fixture
def rng():
    return np.random.default_rng(42)
