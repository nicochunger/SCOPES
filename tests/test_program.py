import pytest

from scopes.scheduler_components import Program


def test_program_initialization(example_instrument):
    program = Program(
        progID="Prog123",
        instrument=example_instrument,
        priority=1,
        time_share_allocated=0.5,
        plot_color="#00FF00",
    )
    assert program.progID == "Prog123"
    assert program.instrument == example_instrument
    assert program.priority == 1
    assert program.time_share_allocated == 0.5
    assert program.plot_color == "#00FF00"


def test_program_default_initialization(example_instrument):
    program = Program(progID="Prog123", instrument=example_instrument, priority=2)
    assert program.progID == "Prog123"
    assert program.instrument == example_instrument
    assert program.priority == 2
    assert program.time_share_allocated == 0.0
    assert program.plot_color is None


def test_program_invalid_progID_type(example_instrument):
    with pytest.raises(TypeError):
        Program(
            progID=123, instrument=example_instrument, priority=1
        )  # progID must be a string


def test_program_invalid_instrument_type():
    with pytest.raises(TypeError):
        Program(
            progID="Prog123", instrument="NotAnInstrument", priority=1
        )  # Instrument must be of type Instrument


def test_program_invalid_priority_value(example_instrument):
    with pytest.raises(ValueError):
        Program(
            progID="Prog123", instrument=example_instrument, priority=5
        )  # Priority must be between 0 and 3
    with pytest.raises(ValueError):
        Program(
            progID="Prog123", instrument=example_instrument, priority=-1
        )  # Priority must be between 0 and 3


def test_program_invalid_priority_type(example_instrument):
    with pytest.raises(TypeError):
        Program(
            progID="Prog123", instrument=example_instrument, priority="High"
        )  # Priority must be an integer


def test_program_invalid_plot_color(example_instrument):
    with pytest.raises(ValueError):
        Program(
            progID="Prog123",
            instrument=example_instrument,
            priority=1,
            plot_color="UX0095",
        )


def test_program_invalid_time_share(example_instrument):
    # Time share must be between 0 and 1
    with pytest.raises(ValueError):
        Program(
            progID="Prog123",
            instrument=example_instrument,
            priority=1,
            time_share_allocated=1.5,
        )
    with pytest.raises(ValueError):
        Program(
            progID="Prog123",
            instrument=example_instrument,
            priority=1,
            time_share_allocated=-0.5,
        )


def test_set_current_time_usage_valid(example_instrument):
    program = Program(
        progID="Prog123",
        instrument=example_instrument,
        priority=1,
        time_share_allocated=0.4,
    )
    program.set_current_time_usage(0.5)
    assert program.time_share_current == 0.5
    assert (program.time_share_pct_diff - 0.1) < 1e-6


def test_set_current_time_usage_invalid(example_instrument):
    # current_time_usage must be between 0 and 1
    program = Program(progID="Prog123", instrument=example_instrument, priority=1)
    with pytest.raises(ValueError):
        program.set_current_time_usage(1.5)
    with pytest.raises(ValueError):
        program.set_current_time_usage(-0.5)


def test_program_str_method(example_instrument):
    program = Program(progID="Prog123", instrument=example_instrument, priority=1)
    expected_str = "Program(\n    ID = Prog123\n    Instrument = Example Spectrograph\n    Time allocated = 0.0\n    Priority = 1)"
    assert str(program) == expected_str


def test_program_repr_method(example_instrument):
    program = Program(progID="Prog123", instrument=example_instrument, priority=1)
    expected_repr = "Program(\n    ID = Prog123\n    Instrument = Example Spectrograph\n    Time allocated = 0.0\n    Priority = 1)"
    assert repr(program) == expected_repr
