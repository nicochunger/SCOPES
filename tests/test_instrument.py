import pytest

from scopes.scheduler_components import Instrument


def test_instrument_initialization():
    instrument = Instrument(
        name="Spectrograph", instrument_type="Spectrograph", plot_color="#FF5733"
    )
    assert instrument.name == "Spectrograph"
    assert instrument.instrument_type == "Spectrograph"
    assert instrument.plot_color == "#FF5733"


def test_instrument_default_initialization():
    instrument = Instrument(name="Camera")
    assert instrument.name == "Camera"
    assert instrument.instrument_type == ""
    assert instrument.plot_color is None


def test_instrument_name_type_error():
    with pytest.raises(TypeError):
        Instrument(name=123)  # Name must be a string


def test_instrument_type_type_error():
    with pytest.raises(TypeError):
        Instrument(name="Spectrograph", instrument_type=123)  # Type must be a string


def test_instrument_invalid_plot_color():
    with pytest.raises(ValueError):
        Instrument(name="Spectrograph", plot_color="UH0095")


def test_instrument_none_plot_color():
    instrument = Instrument(name="Spectrograph", plot_color=None)
    assert instrument.plot_color is None


def test_instrument_str_method():
    instrument = Instrument(name="Spectrograph")
    assert str(instrument) == "Spectrograph"


def test_instrument_repr_method_with_type_and_color():
    instrument = Instrument(
        name="Spectrograph", instrument_type="Spectrograph", plot_color="#FF5733"
    )
    expected_repr = "Instrument(\n    Name = Spectrograph\n    Type = Spectrograph\n    Plot color = #FF5733"
    assert repr(instrument).startswith(expected_repr)


def test_instrument_repr_method_with_type():
    instrument = Instrument(name="Spectrograph", instrument_type="Spectrograph")
    expected_repr = "Instrument(\n    Name = Spectrograph\n    Type = Spectrograph"
    assert repr(instrument).startswith(expected_repr)


def test_instrument_repr_method_without_type_and_color():
    instrument = Instrument(name="Spectrograph")
    expected_repr = "Instrument(\n    Name = Spectrograph"
    assert repr(instrument).startswith(expected_repr)


def test_instrument_equality():
    instrument1 = Instrument(name="Spectrograph", instrument_type="Spectrograph")
    instrument2 = Instrument(name="Spectrograph", instrument_type="Camera")
    instrument3 = Instrument(name="Camera", instrument_type="Spectrograph")
    assert instrument1 == instrument2  # Same name, should be equal
    assert instrument1 != instrument3  # Different names, should not be equal
    assert instrument1 != "Not an Instrument"  # Should not be equal to a different type
