from datetime import date

import pytest
from astroplan import Observer

from scopes.scheduler_components import Night

observer = Observer.at_site("lasilla")


def test_init_invalid_observations_within():
    with pytest.raises(ValueError):
        Night(night_date=date.today(), observations_within="invalid", observer=observer)


def test_init_civil_observations_within():
    n = Night(night_date=date.today(), observations_within="civil", observer=observer)
    assert n.observations_within == "civil"
    assert n.obs_within_limits[0] == n.civil_evening
    assert n.obs_within_limits[1] == n.civil_morning
    assert len(n.night_time_range) == 300


def test_init_nautical_observations_within():
    n = Night(
        night_date=date.today(), observations_within="nautical", observer=observer
    )
    assert n.observations_within == "nautical"
    assert n.obs_within_limits[0] == n.nautical_evening
    assert n.obs_within_limits[1] == n.nautical_morning
    assert len(n.night_time_range) == 300


def test_init_astronomical_observations_within():
    n = Night(
        night_date=date.today(), observations_within="astronomical", observer=observer
    )
    assert n.observations_within == "astronomical"
    assert n.obs_within_limits[0] == n.astronomical_evening
    assert n.obs_within_limits[1] == n.astronomical_morning
    assert len(n.night_time_range) == 300
