import uuid

import pytest

from scopes.scheduler_components import Night, Observation


def test_observation_initialization(example_target):
    # Test with default instrument (None)
    observation_default = Observation(target=example_target, duration=3600)
    assert observation_default.target == example_target
    assert observation_default.duration == 3600 / 86400  # exposure time in days
    assert observation_default.instrument is None  # Check default instrument
    assert observation_default.tel_alt_lower_lim == 10.0
    assert observation_default.tel_alt_upper_lim == 90.0
    assert observation_default.score == 0.0
    assert isinstance(observation_default.unique_id, uuid.UUID)

    # Test with specified instrument
    instrument_name = "TestSpec"
    observation_instrument = Observation(
        target=example_target, duration=3600, instrument=instrument_name
    )
    assert (
        observation_instrument.instrument == instrument_name
    )  # Check specified instrument


def test_observation_invalid_target_type():
    with pytest.raises(TypeError):
        Observation(
            target="InvalidTarget", duration=3600, instrument="TestSpec"
        )  # Added instrument


def test_observation_invalid_duration_value(example_target):
    with pytest.raises(ValueError):
        Observation(
            target=example_target, duration=0, instrument="TestSpec"
        )  # Added instrument


def test_observation_invalid_instrument_type(example_target):
    with pytest.raises(TypeError):
        Observation(
            target=example_target, duration=3600, instrument=123
        )  # Invalid instrument type


def test_observation_warning_for_short_duration(example_target):
    with pytest.warns(UserWarning):
        Observation(
            target=example_target,
            duration=0.5,
            instrument="TestSpec",  # Added instrument
        )  # Warning for exposure time < 1 second


def test_observation_warning_for_long_duration(example_target):
    with pytest.warns(UserWarning):
        Observation(
            target=example_target,
            duration=32401,
            instrument="TestSpec",  # Added instrument
        )  # Warning for exposure time > 9 hours


def test_observation_set_night(example_observation: Observation, example_night):
    example_observation.set_night(example_night)
    assert example_observation.night == example_night


def test_observation_set_start_time(example_observation: Observation):
    start_time = 2456789.5  # Example Julian Date
    example_observation.set_start_time(start_time)
    assert example_observation.start_time == start_time
    assert example_observation.end_time == start_time + example_observation.duration


def test_observation_skypath(example_observation: Observation, example_night: Night):
    example_observation.set_night(example_night)
    example_observation.set_start_time(example_night.night_time_range[0].jd)
    example_observation.skypath()
    assert hasattr(example_observation, "night_altitudes")
    assert hasattr(example_observation, "night_azimuths")
    assert hasattr(example_observation, "night_airmasses")
    assert hasattr(example_observation, "min_altitude")
    assert hasattr(example_observation, "max_altitude")
    assert hasattr(example_observation, "culmination_time")
    assert hasattr(example_observation, "rise_time")
    assert hasattr(example_observation, "set_time")


def test_observation_fairness_no_merits(example_observation: Observation):
    example_observation.target.fairness_merits.clear()  # Ensure no fairness merits
    assert example_observation.fairness() == 1.0  # No merits, fairness should be 1.0


def test_observation_fairness_with_merits(
    example_observation_with_fairness: Observation,
):
    assert (
        example_observation_with_fairness.fairness() == 1.2
    )  # Should match the merit function's return value


def test_observation_update_alt_airmass(
    example_observation: Observation, example_night: Night
):
    example_observation.set_night(example_night)
    example_observation.set_start_time(example_night.night_time_range[0].jd)
    example_observation.skypath()  # Must be run to initialize skypath data
    example_observation.update_alt_airmass()
    assert len(example_observation.obs_time_range) > 0
    assert len(example_observation.obs_altitudes) > 0
    assert len(example_observation.obs_azimuths) > 0
    assert len(example_observation.obs_airmasses) > 0


def test_observation_feasible_no_merits(example_observation: Observation):
    example_observation.target.veto_merits.clear()  # Ensure no veto merits
    assert (
        example_observation.feasible() == 1.0
    )  # No veto merits, sensibility should be 1.0


def test_observation_feasible_with_veto_merit(
    example_observation_with_veto: Observation,
):
    assert (
        example_observation_with_veto.feasible() == 0.0
    )  # Should match the veto merit function's return value


def test_observation_evaluate_score_with_fairness(
    example_observation_with_fairness: Observation,
):
    example_observation_with_fairness.feasible()  # sensibility must be calculated first
    score = example_observation_with_fairness.evaluate_score()
    assert (
        score == 1.2
    )  # Since fairness = 1.2, sensibility = 1, efficiency = 1 (default)


def test_observation_evaluate_score_with_efficiency(
    example_observation_with_efficiency: Observation,
):
    example_observation_with_efficiency.set_start_time(2456789.5)
    example_observation_with_efficiency.feasible()  # sensibility must be calculated first
    score = example_observation_with_efficiency.evaluate_score()
    assert (
        score == 0.9
    )  # Since fairness = 1 (default), sensibility = 1, efficiency = 0.9


def test_observation_evaluate_score_with_veto(
    example_observation_with_veto: Observation,
):
    example_observation_with_veto.set_start_time(2456789.5)
    example_observation_with_veto.feasible()  # sensibility must be calculated first
    score = example_observation_with_veto.evaluate_score()
    assert score == 0.0  # Since veto merits should zero out the score


def test_observation_update_start_and_score(
    example_observation: Observation, example_night: Night
):
    start_time = 2460524.7  # Example Julian Date where the example target is visible
    example_observation.set_night(example_night)
    example_observation.set_start_time(start_time - 0.1)
    example_observation.skypath()
    example_observation.update_start_and_score(start_time)
    assert example_observation.start_time == start_time
    assert example_observation.score > 0.0  # Assume some score is calculated


def test_observation_str_method(example_observation: Observation):
    example_observation.set_start_time(2456789.5)
    # Fixture provides 'Example Spectrograph' as instrument
    instrument_str = f"            Instrument: {example_observation.instrument},\n"
    expected_str = (
        f"Observation(Target: {example_observation.target.name},\n"
        f"{instrument_str}"  # Added instrument string
        f"            Start time: {example_observation.start_time},\n"
        f"            Exposure time: {example_observation.duration},\n"
        f"            Score: {example_observation.score})"
    )
    assert str(example_observation) == expected_str


def test_observation_repr_method(example_observation: Observation):
    example_observation.set_start_time(2456789.5)
    # Fixture provides 'Example Spectrograph' as instrument
    instrument_str = f"            Instrument: {example_observation.instrument},\n"
    expected_repr = (
        f"Observation(Target: {example_observation.target.name},\n"
        f"{instrument_str}"  # Added instrument string
        f"            Start time: {example_observation.start_time},\n"
        f"            Exposure time: {example_observation.duration},\n"
        f"            Score: {example_observation.score})"
    )
    assert repr(example_observation) == expected_repr


def test_observation_equality(example_target):
    observation1 = Observation(
        target=example_target, duration=3600, instrument="Spec1"
    )  # Added instrument
    observation2 = Observation(
        target=example_target, duration=3600, instrument="Spec1"
    )  # Added instrument
    observation3 = Observation(
        target=example_target, duration=3600, instrument="Spec2"
    )  # Different instrument
    assert observation1 != observation2  # Different unique IDs, should not be equal
    assert observation1 == observation1  # Same object, should be equal
    # Equality is based on unique_id, not instrument name
    assert observation1 != observation3


def test_observation_equality_different_type(example_observation):
    assert (
        example_observation != "Not an Observation"
    )  # Should not be equal to a different type
