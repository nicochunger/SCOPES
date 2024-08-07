import pytest

from scopes.scheduler_components import Overheads


def test_overheads_initialization():
    overheads = Overheads(slew_rate_az=1.0, slew_rate_alt=0.5, cable_wrap_angle=180.0)
    assert overheads.slew_rate_az == 1.0
    assert overheads.slew_rate_alt == 0.5
    assert overheads.cable_wrap_angle == 180.0
    assert overheads.overheads == []


def test_overheads_invalid_slew_rate():
    with pytest.raises(ValueError):
        Overheads(slew_rate_az=-1.0, slew_rate_alt=0.5)  # Invalid slew rate


def test_overheads_invalid_cable_wrap_angle():
    with pytest.raises(ValueError):
        Overheads(
            slew_rate_az=1.0, slew_rate_alt=0.5, cable_wrap_angle=400.0
        )  # Invalid angle


def test_overheads_validate_function_params():
    overheads = Overheads(slew_rate_az=1.0, slew_rate_alt=0.5)

    def valid_func(observation1, observation2):
        pass

    def invalid_func(obs1, obs2):
        pass

    assert overheads._validate_function_params(valid_func)
    assert not overheads._validate_function_params(invalid_func)


def test_overheads_is_angle_between():
    overheads = Overheads(slew_rate_az=1.0, slew_rate_alt=0.5)
    assert overheads._is_angle_between(10, 30, 20)
    assert overheads._is_angle_between(350, 10, 0)
    assert not overheads._is_angle_between(30, 10, 40)


def test_overheads_add_overhead():
    overheads = Overheads(slew_rate_az=1.0, slew_rate_alt=0.5)

    def valid_func(observation1, observation2):
        return 100

    overheads.add_overhead(valid_func)
    assert len(overheads.overheads) == 1


def test_overheads_add_overhead_invalid_function():
    overheads = Overheads(slew_rate_az=1.0, slew_rate_alt=0.5)

    def invalid_func(obs1, obs2):
        return 100

    with pytest.raises(ValueError):
        overheads.add_overhead(invalid_func)


def test_overheads_calculate_slew_time(
    example_full_observation1, example_full_observation2, example_overheads: Overheads
):
    slew_time = example_overheads.calculate_slew_time(
        example_full_observation1, example_full_observation2
    )
    assert slew_time > 0


def test_overheads_calculate_transition(
    example_full_observation1, example_full_observation2, example_overheads: Overheads
):
    def custom_overhead(observation1, observation2):
        return 100  # Mocking overhead time in seconds

    example_overheads.add_overhead(custom_overhead)
    total_overhead = example_overheads.calculate_transition(
        example_full_observation1, example_full_observation2
    )
    assert total_overhead > 100 / 86400  # Overhead time in days


def test_overheads_str_method():
    overheads = Overheads(slew_rate_az=1.0, slew_rate_alt=0.5, cable_wrap_angle=180.0)
    expected_str = "Overheads(\n\tslew_rate_az=1.0, \n\tslew_rate_alt=0.5, \n\tcable_wrap_limit=180.0)"
    assert str(overheads) == expected_str
