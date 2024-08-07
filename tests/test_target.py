import pytest

from scopes.scheduler_components import Target


def test_target_initialization(example_target: Target, example_program, example_coords):
    assert example_target.name == "Example Target"
    assert example_target.program == example_program
    assert example_target.coords == example_coords
    assert example_target.ra_deg == example_coords.ra.deg
    assert example_target.dec_deg == example_coords.dec.deg
    assert example_target.priority == 1
    assert example_target.comment == "This is a test target"


def test_target_default_initialization(example_program, example_coords):
    target = Target(
        name="Example Target", program=example_program, coords=example_coords
    )
    assert target.name == "Example Target"
    assert target.program == example_program
    assert target.coords == example_coords
    assert target.priority is None
    assert target.comment == ""


def test_target_invalid_name_type(example_program, example_coords):
    with pytest.raises(TypeError):
        Target(
            name=123, program=example_program, coords=example_coords
        )  # Name must be a string


def test_target_invalid_program_type(example_coords):
    with pytest.raises(TypeError):
        Target(
            name="Example Target", program="InvalidProgram", coords=example_coords
        )  # Program must be of type Program


def test_target_invalid_coords_type(example_program):
    with pytest.raises(TypeError):
        Target(
            name="Example Target", program=example_program, coords="InvalidCoords"
        )  # Coords must be of type SkyCoord


def test_target_invalid_priority_value(example_program, example_coords):
    with pytest.raises(ValueError):
        Target(
            name="Example Target",
            program=example_program,
            coords=example_coords,
            priority=5,
        )  # Priority must be between 0 and 3


def test_target_invalid_priority_type(example_program, example_coords):
    with pytest.raises(TypeError):
        Target(
            name="Example Target",
            program=example_program,
            coords=example_coords,
            priority="High",
        )  # Priority must be an integer


def test_target_invalid_comment_type(example_program, example_coords):
    with pytest.raises(TypeError):
        Target(
            name="Example Target",
            program=example_program,
            coords=example_coords,
            comment=123,
        )  # Comment must be a string


def test_target_add_merit(example_target, example_merit):
    example_target.add_merit(example_merit)
    assert example_merit in example_target.veto_merits


def test_target_add_merit_invalid_type(example_target):
    with pytest.raises(TypeError):
        example_target.add_merit("InvalidMerit")  # Merit must be of type Merit


def test_target_add_merits(example_target, example_merit):
    merits = [example_merit]
    example_target.add_merits(merits)
    assert example_merit in example_target.veto_merits


def test_target_add_merits_invalid_list_type(example_target):
    with pytest.raises(TypeError):
        example_target.add_merits("InvalidMerits")  # Merits must be a list


def test_target_add_merits_invalid_item_type(example_target):
    with pytest.raises(TypeError):
        example_target.add_merits(
            ["InvalidMerit"]
        )  # All items in merits list must be of type Merit


def test_target_str_method(example_target):
    assert str(example_target).startswith("Target(Name: Example Target")


def test_target_repr_method(example_target):
    assert repr(example_target).startswith("Target(Name: Example Target")


def test_target_equality(example_target, example_program, example_coords):
    target1 = Target(
        name="Example Target",
        program=example_program,
        coords=example_coords,
        priority=1,
    )
    target2 = Target(
        name="Example Target",
        program=example_program,
        coords=example_coords,
        priority=2,
    )
    target3 = Target(
        name="Different Target",
        program=example_program,
        coords=example_coords,
        priority=1,
    )
    assert target1 == target2  # Same name, should be equal
    assert target1 != target3  # Different name, should not be equal


def test_target_equality_different_type(example_target):
    assert example_target != "Not a Target"  # Should not be equal to a different type
