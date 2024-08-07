from datetime import date

import pytest
from astroplan import Observer
from astropy.time import Time

from scopes.scheduler_components import Night


def test_night_initialization(example_night):
    assert example_night.night_date == date(2024, 8, 1)
    assert example_night.observations_within == "astronomical"
    assert isinstance(example_night.observer, Observer)


def test_invalid_observations_within(example_observer):
    with pytest.raises(ValueError):
        Night(
            night_date=date(2024, 8, 1),
            observations_within="invalid",
            observer=example_observer,
        )


def test_solar_midnight_calculation(example_night):
    assert isinstance(example_night.solar_midnight, Time)


def test_twilight_times(example_night):
    assert isinstance(example_night.sunset, Time)
    assert isinstance(example_night.sunrise, Time)
    assert isinstance(example_night.civil_evening, float)
    assert isinstance(example_night.nautical_evening, float)
    assert isinstance(example_night.astronomical_evening, float)
    assert isinstance(example_night.civil_morning, float)
    assert isinstance(example_night.nautical_morning, float)
    assert isinstance(example_night.astronomical_morning, float)


def test_night_time_range(example_night):
    assert len(example_night.night_time_range) == 300


# def test_culmination_window(example_night):
#     TODO: Create test Observations
#     obs1 = Observation()
#     obs2 = Observation()

#     obs_list = [obs1, obs2]
#     example_night.calculate_culmination_window(obs_list)
#     assert example_night.culmination_window == (2456789.5, 2456789.5)


def test_str_representation(example_night):
    assert str(example_night).startswith("Night(Date: 2024-08-01,")


def test_night_equality(example_night):
    other_night = Night(
        night_date=date(2024, 8, 1),
        observations_within="astronomical",
        observer=example_night.observer,
    )
    assert example_night == other_night
    different_night = Night(
        night_date=date(2024, 8, 2),
        observations_within="astronomical",
        observer=example_night.observer,
    )
    assert example_night != different_night
