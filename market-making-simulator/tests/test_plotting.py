"""Chart smoke tests and palette invariants.

``plotting`` is excluded from the coverage target, but the figures are in the README, so a
silent breakage would ship. These render every one against real simulation output.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from mmsim import plotting, style
from mmsim.avellaneda_stoikov import AvellanedaStoikovParams
from mmsim.calibration import estimate_fill_intensity
from mmsim.engine import simulate_book
from mmsim.experiments import build_policy_set, compare_policies_reference, sensitivity_sweep
from mmsim.flow import FlowConfig
from mmsim.strategies import AvellanedaStoikovPolicy, SymmetricPolicy
from mmsim.types import MarketConfig, Side


def test_categorical_palette_is_the_validated_four() -> None:
    assert style.CATEGORICAL == ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")


def test_ordinal_colors_run_light_to_dark() -> None:
    colors = style.ordinal_colors(5)

    def luminance(hex_color: str) -> float:
        r, g, b = matplotlib.colors.to_rgb(hex_color)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lums = [luminance(c) for c in colors]
    assert all(a > b for a, b in itertools.pairwise(lums))


def test_geometric_detection() -> None:
    import numpy as np

    from mmsim.plotting import _looks_geometric

    assert _looks_geometric(np.geomspace(0.01, 3.0, 9))
    assert not _looks_geometric(np.linspace(0.5, 5.0, 9))
    assert not _looks_geometric(np.array([1.0, 2.0]))
    assert not _looks_geometric(np.array([-1.0, 1.0, 3.0]))


@pytest.fixture(scope="module")
def comparison():
    params = AvellanedaStoikovParams(
        gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0, horizon=1.0
    )
    return compare_policies_reference(
        build_policy_set(params, inventory_limit=3.0), params, n_runs=25, n_steps=100
    )


def test_model_figures_render(paper_params: AvellanedaStoikovParams) -> None:
    fig = plotting.plot_reservation_price(paper_params)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_comparison_figures_render(comparison) -> None:
    for fig in (
        plotting.plot_inventory_paths(comparison, n_paths=5),
        plotting.plot_pnl_distribution(comparison),
        plotting.plot_risk_return(comparison),
    ):
        assert len(fig.axes) >= 1
        plt.close(fig)


def test_session_and_markout_figures_render(
    informed_flow: FlowConfig, market: MarketConfig
) -> None:
    runs = [
        simulate_book(SymmetricPolicy(half_spread=0.04), informed_flow, market, n_steps=400, seed=s)
        for s in range(4)
    ]
    for fig in (
        plotting.plot_quotes_and_inventory(runs[0]),
        plotting.plot_markout({"Symmetric": runs}, horizons=(1, 5, 25)),
    ):
        assert len(fig.axes) >= 1
        plt.close(fig)


def test_fill_intensity_figure_renders(quiet_flow: FlowConfig, market: MarketConfig) -> None:
    fit = estimate_fill_intensity(quiet_flow, market, n_steps=800, seed=1)
    fig = plotting.plot_fill_intensity_fit(fit, market)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_volatility_signature_figure_renders(market: MarketConfig) -> None:
    from mmsim.calibration import volatility_signature

    signatures = {
        "quiet": volatility_signature(
            FlowConfig(informed_fraction=0.0), market, n_steps=2000, seed=1
        ),
        "toxic": volatility_signature(
            FlowConfig(informed_fraction=0.2), market, n_steps=2000, seed=1
        ),
    }
    fig = plotting.plot_volatility_signature(signatures)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_sensitivity_figure_renders() -> None:
    import numpy as np

    world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)

    def build(value: float):
        params = AvellanedaStoikovParams(
            gamma=value, kappa=1.5, arrival_rate=140.0, sigma=2.0, horizon=1.0
        )
        return AvellanedaStoikovPolicy(params), world

    sweeps = {
        "gamma": sensitivity_sweep(
            list(np.geomspace(0.02, 2.0, 4)), build, "gamma", n_runs=20, n_steps=80
        )
    }
    fig = plotting.plot_sensitivity(sweeps)
    assert fig.axes[0].get_xscale() == "log"
    plt.close(fig)


def test_book_snapshot_figure_renders(market: MarketConfig) -> None:
    fig = plotting.plot_book_snapshot([(99, 3.0), (98, 5.0)], [(101, 4.0)], market)
    assert len(fig.axes) >= 1
    plt.close(fig)
    empty = plotting.plot_book_snapshot([], [], market)
    plt.close(empty)


def test_save_all_writes_png(tmp_path, paper_params: AvellanedaStoikovParams) -> None:
    fig = plotting.plot_reservation_price(paper_params)
    written = plotting.save_all({"reservation": fig}, tmp_path / "out")
    assert len(written) == 1 and written[0].exists()
    assert written[0].stat().st_size > 1000


def test_side_enum_used_by_plots() -> None:
    assert Side.BID.label == "bid"
