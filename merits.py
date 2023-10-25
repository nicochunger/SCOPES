""" Definition of all the generic merit functions """

import astropy.units as u
import numpy as np
from astropy.time import Time, TimeDelta

import config
from class_definitions import Observation


def at_night(observation: Observation, which: str) -> float:
    """Merit function that returns 1 if the observation is at night, 0 otherwise"""
    valid_options = ["civil", "nautical", "astronomical"]
    start_time = observation.start_time
    end_time = observation.start_time + observation.exposure_time
    if which == "civil":
        return (start_time > observation.night.civil_evening) and (
            end_time < observation.night.civil_morning
        )
    elif which == "nautical":
        return (start_time > observation.night.nautical_evening) and (
            end_time < observation.night.nautical_morning
        )
    elif which == "astronomical":
        return (start_time > observation.night.astronomical_evening) and (
            end_time < observation.night.astronomical_morning
        )
    else:
        raise ValueError(f"which must be one of {valid_options}")


def airmass(
    observation: Observation, max: float, alpha: float = 1.0, verbose: bool = False
) -> float:
    """
    Merit function on the airmass of the observation. It uses a hyperbolic tangent function to
    gradually increase the merit as the airmass increases. The specific shape can be set with the
    parameters min and alpha. The exact formula is:
    # TODO Implement the hyperbolic tangent function so that there is some leeway in the airmass
    """
    # Claculate airmass throughout the exposure of the observation
    airmasses = observation.coords_altaz.secz

    if verbose:
        print(f"Airmasses: {airmasses}")
        print(f"Max airmass: {airmasses.max()}")
    # Evaluate maximum airmass of the observation and check if it is below the maximum and return 0.
    # Otherwise return the merit: (1/secz(max)**alpha)
    if airmasses.max() > max:
        if verbose:
            print("Airmass higher than maximum, return 0.")
        return 0.0
    else:
        if verbose:
            print("Airmass within limits, return merit.")
        return 1 / airmasses.max() ** alpha


def altitude(
    observation: Observation,
    min: float = config.scheduling_defaults["telescope elevation limits"][0],  # type: ignore
    max: float = config.scheduling_defaults["telescope elevation limits"][1],  # type: ignore
) -> float:  
    """Altitude constraint merit function"""
    # Claculate altitude throughout the exposure of the observation
    range_altitudes = observation.coords_altaz.alt.deg

    if range_altitudes.min() < min or range_altitudes.max() > max:
        return 0.0
    else:
        return 1.0


def culmination(observation: Observation, verbose: bool = False) -> float:
    """Culmination constraint merit function.
    This merit calculates the current height of the target and the proportion to the maximum height
     it will reach during the current night."""
    # Calculate the current altitude of the target
    current_altitude = observation.coords_altaz.alt.deg[0]
    # Calculate altitude proportional to available altitue range
    altitude_prop = (current_altitude - observation.min_altitude) / (
        observation.max_altitude - observation.min_altitude
    )
    if verbose:
        print(f"Current altitude: {current_altitude}")
        print(f"Max altitude: {observation.max_altitude}")
        print(f"Min altitude: {observation.min_altitude}")
        print(f"Altitude proportion: {altitude_prop}")
    return altitude_prop


def egress(observation: Observation, verbose: bool = False) -> float:
    """Egress constraint merit function"""
    if (observation.start_time < observation.rise_time) or (
        observation.start_time > observation.set_time
    ):
        if verbose:
            print("Observation starts before rise time or after set time, return 0.")
        return 0.0
    else:
        # Claculate altitude throughout the exposure of the observation
        range_observable = observation.set_time - observation.rise_time
        observable_range_prop = (
            observation.start_time - observation.rise_time
        ) / range_observable
        if verbose:
            print(f"Current time: {observation.start_time}")
            print(f"First time: {observation.rise_time}")
            print(f"Last time: {observation.set_time}")
            print(f"Observable range proportion: {observable_range_prop}")
        return observable_range_prop


def cadence(
    observation: Observation, delay: TimeDelta, alpha: float, beta: float = 0.5
) -> float:
    """
    Calcualtes the merit for the desired cadence of observations.
    It uses a piecewise function depending if the target is under or over the cadence delay.
    If it hasn't reached the cadence yet, it uses a hyperbolic tanget function to slowly increase
    the merit up to 0.5 when it reached the cadence. Once its over the cadence its a root function
    that continuously increases as the observation is more overdue. The specific attack and release
    shapes can be set with the parameters alpha and beta.
    First the percent overdue is calculated by simply looking at how many days the target is before
    or after the desired cadence as a proportion of the cadence. Then the exact formulas are:

    When target is underdue:
    0.5*(tanh(pct_overdue/alpha)+1)

    When target is overdue:
    0.5+(pct_overdue/50*alpha)**beta


    Args:
        observation (Observation): The Observation object to be used
        delay (int): The value for the cadence in days
        alpha (float): The attack parameter for when the target is underdue
        beta (float): The release parameter for when the target is overdue

    Returns:
        score (float): The score of this merit
    """
    pct_overdue = (observation.start_time - (observation.target.last_obs + delay)).to(
        u.day
    ).value / delay.to(u.day).value
    if pct_overdue <= 0:
        # Target has not reacehd the desired cadence yet
        # Use hyperbolic tangent to gradually increase merit as it approaches the set cadence
        return 0.5 * (np.tanh(pct_overdue / alpha) + 1)
    else:
        # Target has reacehd the cadence and is overdue
        # Use a root function to slowly keep increasing the merit as the observation is increasingly overdue
        return 0.5 + (pct_overdue / (50 * alpha)) ** beta


def moon_distance(observation: Observation, min: float = 30.0) -> float:
    """Moon distance constraint merit function"""
    # Create the AltAz frame for the moon
    moon_altaz_frame = observation.observer.moon_altaz(time=observation.time_array)
    # Claculate moon distance throughout the exposure of the observation
    range_moon_distance = observation.target.coords.separation(moon_altaz_frame).deg

    # TODO: Implement a merit that takes into account the moon illumination
    if range_moon_distance.min() < min:
        return 0.0
    else:
        return 1.0


# def moon_radial_velocity(observation: Observation, max: float = 100.0) -> float:
#     """ Moon radial velocity constraint merit function """
#     # Create the AltAz frame for the moon
#     moon_altaz_frame = observation.observer.moon_altaz(time=observation.time_array)
# TODO check this. How to calcualte the moon rv. This was recommended by copilot.
# The attribute radial_velocity in AltAz object does inded exist, but Im not sure how
# to set it correctly.
# Claculate moon radial velocity throughout the exposure of the observation
# range_moon_radial_velocity = observation.target.radial_velocity - moon_altaz_frame.radial_velocity

# if range_moon_radial_velocity.max() > max:
#     return 0.0
# else:
#     return 1.0


def gaussian(x, x0, P, s):
    """
    Gaussian merit function.

    Analytic expression: exp(-0.5(x/P)^2/s^2)

    Parameters
    ----------
    x : float
        The x value at which to evaluate the merit function.
    x0 : float
        Where the peak of the merit will be centered
    P : float
        The period of the Gaussian.
    s : float
        The standard deviation of the Gaussian.
    """

    return np.exp(-0.5 * (x / P) ** 2 / s**2)


def periodic_gaussian(x, x0, P, s):
    """
    Periodic Gaussian merit function.

    Analytic expression: exp(-0.5(sin(2pi(x-x0)/P)/s)^2)

    The introduction of the sine function breaks the traditional meaning of the standard deviation. So the
    s parameter will have to be finetuned depending on the period.

    Parameters
    ----------
    x : float
        The x value at which to evaluate the merit function.
    x0 : float
        Where the peak of the merit will be centered
    P : float
        The period of the Gaussian.
    s : float
        Measure of the width of each Gaussian.
    """

    return np.exp(-0.5 * (np.sin(2 * np.pi * (x - x0) / P) / s) ** 2)


def periodic_box(x, x0, P, width):
    """
    Periodic box merit function.

    It's just a box function that repeats with perio P.
    Parameters
    ----------
    x : float
        The x value at which to evaluate the merit function.
    x0 : float
        Where the peak of the merit will be centered
    P : float
        The period of the box functions.
    width : float
        The width of each box.
    """

    # TODO: Test if this is correct
    offset = (x - x0) % P
    return offset > -width / 2 or offset < width / 2
