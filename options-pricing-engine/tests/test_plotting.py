"""Chart smoke tests and palette invariants.

``plotting`` is excluded from the coverage target: asserting on pixels is brittle and
asserting "it did not raise" is coverage theatre. What *is* worth testing is that every
figure renders without error on a real surface (they are in the README, so a silent
breakage would ship), and that the palette obeys the rules the style module claims.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from optpricing import calibration as cal
from optpricing import plotting, style
from optpricing.heston import HestonParams


def test_categorical_palette_is_the_validated_four() -> None:
    """The exact hexes that cleared the all-pairs colour-vision gates. Not decoration."""
    assert style.CATEGORICAL == ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
    assert len(set(style.CATEGORICAL)) == 4


def test_ordinal_colors_run_light_to_dark() -> None:
    colors = style.ordinal_colors(6)
    assert len(colors) == 6
    assert len(set(colors)) == 6

    def luminance(hex_color: str) -> float:
        rgb = matplotlib.colors.to_rgb(hex_color)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    lums = [luminance(c) for c in colors]
    assert all(a > b for a, b in itertools.pairwise(lums))
    assert style.ordinal_colors(0) == []
    assert len(style.ordinal_colors(1)) == 1


def test_house_style_applies_and_disables_top_right_spines() -> None:
    style.apply_house_style()
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["legend.frameon"] is False
    cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    assert tuple(cycle) == style.CATEGORICAL


@pytest.mark.parametrize(
    "factory",
    [
        lambda: plotting.plot_binomial_convergence(max_steps=40),
        lambda: plotting.plot_exercise_boundary(steps=60),
        lambda: plotting.plot_greeks_panel(),
        lambda: plotting.plot_mc_convergence(n_replications=3),
    ],
)
def test_model_figures_render(factory) -> None:
    fig = factory()
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_surface_figures_render(synthetic_surface: pd.DataFrame) -> None:
    for fig in (
        plotting.plot_smiles(synthetic_surface),
        plotting.plot_term_structure(synthetic_surface),
        plotting.plot_surface_heatmap(synthetic_surface),
        plotting.plot_surface_3d(synthetic_surface),
        plotting.plot_smile_grid(synthetic_surface),
    ):
        assert len(fig.axes) >= 1
        plt.close(fig)


def test_smile_grid_has_one_visible_panel_per_expiry(synthetic_surface: pd.DataFrame) -> None:
    fig = plotting.plot_smile_grid(synthetic_surface, ncols=4)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == synthetic_surface["tau"].nunique()
    plt.close(fig)


def test_parity_and_greeks_fd_figures_render(
    synthetic_snapshot, flat_rate_curve, synthetic_surface
) -> None:
    from optpricing import data as data_mod
    from optpricing import greeks_check, parity

    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    fwd = data_mod.implied_forward_curve(clean, synthetic_snapshot.spot, flat_rate_curve)
    res = parity.run_parity_analysis(
        synthetic_snapshot, clean, flat_rate_curve, fwd, synthetic_surface, steps=30
    )
    fig = plotting.plot_parity_residuals(res.residuals)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == fwd["tau"].nunique()
    plt.close(fig)

    points = greeks_check.default_grid(taus=(0.1, 1.0), sigmas=(0.2,))
    errors = greeks_check.error_table(points)
    sweeps = {"gamma": greeks_check.step_sweep(points[3], "gamma", multipliers=[1e-4, 1e-3])}
    fig = plotting.plot_greeks_fd(errors, sweeps, {"gamma": 1e-3}, (1e-6, 5e-3))
    assert len(fig.axes) == 2
    plt.close(fig)


def test_heston_fit_figures_render(synthetic_surface: pd.DataFrame) -> None:
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6)
    errors = cal.surface_errors(params, synthetic_surface, 500.0, exact_vols=True)
    for fig in (
        plotting.plot_heston_fit(errors, max_expiries=3),
        plotting.plot_heston_fit_errors(errors, params),
    ):
        assert len(fig.axes) >= 1
        plt.close(fig)


def test_save_all_writes_png_files(tmp_path) -> None:
    fig = plotting.plot_greeks_panel()
    written = plotting.save_all({"greeks": fig}, tmp_path / "out")
    assert len(written) == 1
    assert written[0].exists()
    assert written[0].suffix == ".png"
    assert written[0].stat().st_size > 1000
