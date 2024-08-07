import pytest

from scopes.scheduler_components import Merit


@pytest.fixture
def example_merit_func():
    def custom_merit_function(observation, example_param, verbose=False):
        """Example merit function."""
        if verbose:
            print("Custom merit function called.")
        return example_param

    return custom_merit_function


def test_merit_initialization(example_merit_func):
    merit = Merit(
        name="Example Merit",
        func=example_merit_func,
        merit_type="efficiency",
        parameters={"example_param": 0.5},
    )
    assert merit.name == "Example Merit"
    assert merit.func == example_merit_func
    assert merit.description == "Example merit function."
    assert merit.merit_type == "efficiency"
    assert merit.parameters == {"example_param": 0.5}


def test_merit_invalid_merit_type(example_merit_func):
    with pytest.raises(ValueError):
        Merit(
            name="Invalid Merit",
            func=example_merit_func,
            merit_type="invalid_type",
            parameters={"example_param": 0.5},
        )


def test_merit_invalid_first_parameter_in_func():
    def invalid_merit_function(invalid_param, example_param):
        return example_param

    with pytest.raises(KeyError):
        Merit(
            name="Invalid Merit",
            func=invalid_merit_function,
            merit_type="efficiency",
            parameters={"example_param": 0.5},
        )


def test_merit_missing_required_parameters(example_merit_func):
    with pytest.raises(ValueError):
        Merit(
            name="Missing Params Merit",
            func=example_merit_func,
            merit_type="efficiency",
        )


def test_merit_extra_parameters(example_merit_func):
    with pytest.raises(KeyError):
        Merit(
            name="Extra Params Merit",
            func=example_merit_func,
            merit_type="efficiency",
            parameters={"example_param": 0.5, "extra_param": 1.0},
        )


def test_merit_evaluate_method(example_merit_func, example_observation):
    merit = Merit(
        name="Example Merit",
        func=example_merit_func,
        merit_type="efficiency",
        parameters={"example_param": 0.5},
    )
    result = merit.evaluate(example_observation)
    assert result == 0.5


def test_merit_evaluate_with_runtime_parameters(
    example_merit_func, example_observation
):
    merit = Merit(
        name="Example Merit",
        func=example_merit_func,
        merit_type="efficiency",
        parameters={"example_param": 0.5},
    )
    result = merit.evaluate(example_observation, verbose=True)
    assert result == 0.5


def test_merit_str_method(example_merit_func):
    merit = Merit(
        name="Example Merit",
        func=example_merit_func,
        merit_type="efficiency",
        parameters={"example_param": 0.5},
    )
    expected_str = "Merit(Example Merit, efficiency, {'example_param': 0.5})"
    assert str(merit) == expected_str


def test_merit_repr_method(example_merit_func):
    merit = Merit(
        name="Example Merit",
        func=example_merit_func,
        merit_type="efficiency",
        parameters={"example_param": 0.5},
    )
    expected_repr = "Merit(Example Merit, efficiency, {'example_param': 0.5})"
    assert repr(merit) == expected_repr
