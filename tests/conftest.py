from datetime import date

import pytest
from astroplan import Observer
from astropy.coordinates import Angle, SkyCoord

from scopes.merits import airmass
from scopes.scheduler_components import (
    Instrument,
    Merit,
    Night,
    Observation,
    Overheads,
    Plan,
    Program,
    Target,
)


@pytest.fixture
def example_observer():
    return Observer.at_site("lasilla")


@pytest.fixture
def example_night(example_observer):
    return Night(
        night_date=date(2024, 8, 1),  # National Day of Switzerland
        observations_within="astronomical",
        observer=example_observer,
    )


@pytest.fixture
def example_instrument():
    return Instrument(
        name="Example Spectrograph",
        instrument_type="Spectrograph",
        plot_color="#FF5733",
    )


@pytest.fixture
def example_program(example_instrument):
    return Program(
        progID="Prog123",
        instrument=example_instrument,
        priority=1,
        time_share_allocated=0.3,
    )


@pytest.fixture
def example_merit():
    return Merit(
        name="Example Merit",
        func=airmass,
        merit_type="veto",
        parameters={"limit": 1.8},
    )


@pytest.fixture
def example_custom_merit_func():
    def custom_merit_function(observation, example_param):
        """Example merit function."""
        return 1.0

    return custom_merit_function


@pytest.fixture
def example_coords():
    # A coordinate in the sky for a hypothetical target that is visible during the night
    # of the example_night fixture which is 1st August 2024
    return SkyCoord(ra="20h00m00s", dec="-20d00m00s", frame="icrs")


@pytest.fixture
def example_target(example_program, example_coords):
    target = Target(
        name="Example Target",
        program=example_program,
        coords=example_coords,
        priority=1,
        comment="This is a test target",
    )
    return target


@pytest.fixture
def example_fairness_merit():
    def fairness_merit_func(observation):
        return 1.2  # Example fairness merit score

    return Merit(
        name="Fairness Merit",
        func=fairness_merit_func,
        merit_type="fairness",
        parameters={},
    )


@pytest.fixture
def example_efficiency_merit():
    def efficiency_merit_func(observation):
        return 0.9  # Example efficiency merit score

    return Merit(
        name="Efficiency Merit",
        func=efficiency_merit_func,
        merit_type="efficiency",
        parameters={},
    )


@pytest.fixture
def example_veto_merit():
    def veto_merit_func(observation):
        return 0.0  # Example veto merit score (vetoes the observation)

    return Merit(
        name="Veto Merit", func=veto_merit_func, merit_type="veto", parameters={}
    )


@pytest.fixture
def example_observation(example_target, example_merit):
    example_target.add_merit(merit=example_merit)
    return Observation(target=example_target, exposure_time=600.0)


@pytest.fixture
def example_observation_with_fairness(example_target, example_fairness_merit):
    example_target.add_merit(merit=example_fairness_merit)
    return Observation(target=example_target, exposure_time=600.0)


@pytest.fixture
def example_observation_with_efficiency(example_target, example_efficiency_merit):
    example_target.add_merit(merit=example_efficiency_merit)
    return Observation(target=example_target, exposure_time=600.0)


@pytest.fixture
def example_observation_with_veto(example_target, example_veto_merit):
    example_target.add_merit(merit=example_veto_merit)
    return Observation(target=example_target, exposure_time=600.0)


@pytest.fixture
def example_full_observation1(example_observation: Observation, example_night: Night):
    example_observation.set_night(example_night)
    example_observation.set_start_time(2460524.7)
    example_observation.skypath()
    example_observation.update_start_and_score(2460524.7)
    return example_observation


@pytest.fixture
def example_full_observation2(
    example_coords: SkyCoord, example_night: Night, example_program, example_merit
):
    new_coords = example_coords.directional_offset_by(Angle("1d"), Angle("1d"))
    new_target = Target(
        name="Target002", program=example_program, coords=new_coords, priority=1
    )
    new_target.add_merit(merit=example_merit)
    example_observation2 = Observation(target=new_target, exposure_time=600.0)
    example_observation2.set_night(example_night)
    example_observation2.set_start_time(2460524.7)
    example_observation2.skypath()
    example_observation2.update_start_and_score(2460524.708)
    return example_observation2


@pytest.fixture
def example_full_observation3(
    example_coords: SkyCoord, example_night: Night, example_program, example_merit
):
    new_coords = example_coords.directional_offset_by(Angle("2d"), Angle("2d"))
    new_target = Target(
        name="Target003", program=example_program, coords=new_coords, priority=1
    )
    new_target.add_merit(merit=example_merit)
    example_observation3 = Observation(target=new_target, exposure_time=600.0)
    example_observation3.set_night(example_night)
    example_observation3.set_start_time(2460524.7)
    example_observation3.skypath()
    example_observation3.update_start_and_score(2460524.716)
    return example_observation3


@pytest.fixture
def example_overheads():
    return Overheads(slew_rate_az=1.0, slew_rate_alt=1.0, cable_wrap_angle=180.0)


@pytest.fixture
def example_plan(
    example_full_observation1, example_full_observation2, example_full_observation3
):
    plan = Plan()
    plan.add_observation(example_full_observation1)
    plan.add_observation(example_full_observation2)
    plan.add_observation(example_full_observation3)
    return plan
