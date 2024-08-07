import os
import tempfile
from unittest import mock

import pandas as pd
import pytest

from scopes.scheduler_components import Plan


def test_plan_initialization():
    plan = Plan()
    assert len(plan.observations) == 0
    assert plan.score == 0.0
    assert plan.evaluation == 0.0


def test_plan_add_observation(example_full_observation1):
    plan = Plan()
    plan.add_observation(example_full_observation1)
    assert len(plan.observations) == 1
    assert plan.observations[0] == example_full_observation1


def test_plan_calculate_overhead(example_plan: Plan):
    example_plan.calculate_overhead()
    # 3 observations, each 600 seconds
    assert example_plan.observation_time.total_seconds() == 3 * 600
    # Overhead time should be positive
    assert example_plan.overhead_time.total_seconds() > 0
    # Observation ratio should be positive
    assert example_plan.observation_ratio > 0
    # Overhead ratio should be positive
    assert example_plan.overhead_ratio > 0


def test_plan_calculate_avg_airmass(example_plan: Plan):
    example_plan.calculate_avg_airmass()
    assert example_plan.avg_airmass > 0  # Average airmass should be positive


def test_plan_evaluate_plan(example_plan: Plan):
    evaluation = example_plan.evaluate_plan(w_score=0.4, w_overhead=0.6)
    # Evaluation should be positive
    assert evaluation > 0
    # Plan score should be calculated
    assert example_plan.score > 0
    # Evaluation should match the returned value
    assert example_plan.evaluation == evaluation


def test_plan_evaluate_plan_invalid_weights(example_plan):
    # Weights do not sum to 1
    with pytest.raises(ValueError):
        example_plan.evaluate_plan(w_score=0.5, w_overhead=0.6)


def test_plan_print_stats(example_plan: Plan):
    # This method just prints; no assertion needed
    example_plan.print_stats()


@pytest.mark.parametrize("display", [True, False])
def test_plan_plot(example_plan: Plan, display):
    example_plan.evaluate_plan()
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        with mock.patch("matplotlib.pyplot.close") as mock_close:
            example_plan.plot(display=display)
            if display:
                mock_show.assert_called_once()
            else:
                mock_close.assert_called_once()


@pytest.mark.parametrize("path", [None, "test_plot.png"])
def test_plan_plot_save(example_plan: Plan, path):
    example_plan.evaluate_plan()
    with mock.patch("matplotlib.pyplot.savefig") as mock_savefig:
        example_plan.plot(display=False, path=path)
        if path is not None:
            mock_savefig.assert_called_once_with(path, dpi=300)


def test_plan_plot_interactive(example_plan: Plan):
    with mock.patch("plotly.graph_objs.Figure.show") as mock_show:
        example_plan.plot_interactive()
        mock_show.assert_called_once()


def test_plan_plot_polar(example_plan: Plan):
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        example_plan.plot_polar(display=True)
        mock_show.assert_called_once()


def test_plan_plot_polar_save(example_plan: Plan):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        path = tmpfile.name
        example_plan.plot_polar(display=False, path=path)

    assert os.path.exists(path) and os.path.getsize(path) > 0
    os.remove(path)


def test_plan_plot_altaz(example_plan: Plan):
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        example_plan.plot_altaz(display=True)
        mock_show.assert_called_once()


def test_plan_plot_altaz_save(example_plan: Plan):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        path = tmpfile.name
        example_plan.plot_altaz(display=False, path=path)

    assert os.path.exists(path) and os.path.getsize(path) > 0
    os.remove(path)


def test_plan_to_csv(example_plan: Plan):
    # Create a temporary file to save the CSV output
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
        path = tmpfile.name
        example_plan.to_csv(path)

    # Check that the file is created and is non-empty
    assert os.path.exists(path) and os.path.getsize(path) > 0

    # Clean up the temporary file
    os.remove(path)


def test_plan_to_csv_with_custom_options(example_plan: Plan):
    # Create a temporary file to save the CSV output
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
        path = tmpfile.name
        example_plan.to_csv(path, sep=";", index=False)  # Custom options for to_csv

    # Check that the file is created and is non-empty
    assert os.path.exists(path) and os.path.getsize(path) > 0

    # Read the CSV content with custom options and check
    df_from_csv = pd.read_csv(path, sep=";")
    assert not df_from_csv.empty

    # Clean up the temporary file
    os.remove(path)


def test_plan_len_method(example_plan):
    assert len(example_plan) == 3  # Length should match the number of observations


def test_plan_repr_method(example_plan):
    assert repr(example_plan) == f"<Plan> with {len(example_plan)} observations"


def test_plan_str_method(example_plan: Plan):
    df_str = example_plan.to_df().to_string()
    assert str(example_plan) == df_str
