""" Definition of all the generic merit functions """

# import astropy.units as u
import numpy as np
from astropy.time import TimeDelta

import config
from class_definitions import Observation


def time_share(observation: Observation, alpha: int = 5, beta: float = 5.0) -> float:
    """
    Time share fairness merit. It uses a modified sigmoid function to calculate the merit.
    The specific shape can be set with the parameters alpha and beta. The exact formula is:

    0.5 + (1 / (1 + exp((pct_diff/beta)^alpha)))

    It's shaped in a way so that there is some permissiveness. This means that a program
    can be over or under the allocated time by a certain percentage before its priority is decreased
    or increased. The alpha parameter controls how sharp the sigmoid is, and the beta parameter
    controls how much difference is allowed (in percentage).

    Parameters:
        observation: (Observation)
            The Observation object to be used
        alpha: (float)
            The attack parameter for how sharp the sigmoid is
        beta: (float)
            The leeway parameter for how much difference in time use is allowed
    """
    assert beta > 0.0, "beta for time_share merit must be greater than 0"
    assert alpha > 0, "alpha for time_share merit must be greater than 0"
    assert alpha % 2 == 1, "alpha for time_share merit must be an odd positive integer"
    # Calculate the time share of the observation
    pct_diff = observation.target.program.time_share_pct_diff
    # Calculate the merit
    m = 0.5 + (1 / (1 + np.exp(((pct_diff / beta) ** alpha))))
    return m


def at_night(observation: Observation) -> float:
    """Merit function that returns 1 if the observation is at night, 0 otherwise"""
    start_time = observation.start_time
    end_time = start_time + observation.exposure_time
    if (
        start_time >= observation.night.obs_within_limits[0]
        and end_time <= observation.night.obs_within_limits[1]
    ):
        return 1.0
    else:
        return 0.0


def airmass(observation: Observation, max: float) -> float:
    """
    Merit function on the airmass of the observation. It uses a hyperbolic tangent function to
    gradually increase the merit as the airmass increases. The specific shape can be set with the
    parameters min and alpha. The exact formula is:
    # TODO Implement the hyperbolic tangent function so that there is some leeway in the airmass
    """
    # Claculate airmass throughout the exposure of the observation
    if len(observation.obs_airmasses) == 0:
        return 0.0
    else:
        max_airmass = observation.obs_airmasses.max()
        return max_airmass < max


def altitude(
    observation: Observation,
    min: float = config.scheduling_defaults["telescope elevation limits"][0],  # type: ignore
    max: float = config.scheduling_defaults["telescope elevation limits"][1],  # type: ignore
) -> float:
    """Altitude constraint merit function"""
    # Claculate altitude throughout the exposure of the observation
    if len(observation.obs_altitudes) == 0:
        return 0.0
    else:
        return (observation.obs_altitudes.min() > min) and (observation.obs_altitudes.max() < max)


def culmination(observation: Observation, verbose: bool = False) -> float:
    """Culmination constraint merit function.
    This merit calculates the current height of the target and the proportion to the maximum height
     it will reach during the current night."""
    # Calculate the current altitude of the target
    current_altitude = observation.obs_altitudes[0]
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
        observable_range_prop = (observation.start_time - observation.rise_time) / range_observable
        if verbose:
            print(f"Current time: {observation.start_time}")
            print(f"First time: {observation.rise_time}")
            print(f"Last time: {observation.set_time}")
            print(f"Observable range proportion: {observable_range_prop}")
        return observable_range_prop


def cadence(
    observation: Observation,
    delay: TimeDelta,
    alpha: float,
    beta: float = 0.5,
    verbose: bool = False,
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
    if verbose:
        print(f"{observation.start_time = }")
        print(f"{observation.target.last_obs = }")
        print(f"{delay.value = }")
    pct_overdue = (
        observation.start_time - (observation.target.last_obs + delay.value)
    ) / delay.value
    if pct_overdue <= 0:
        # Target has not reacehd the desired cadence yet
        # Use hyperbolic tangent to gradually increase merit as it approaches the set cadence
        return 0.5 * (np.tanh(pct_overdue / alpha) + 1)
    else:
        # Target has reacehd the cadence and is overdue
        # Use a root function to slowly keep increasing the merit as the observation is increasingly overdue
        return 0.5 + (pct_overdue / (50 * alpha)) ** beta


def periodic_gaussian(
    observation: Observation,
    epoch: float,
    period: float,
    sigma: float,
    phases: list = [0.0],
    verbose: bool = False,
) -> float:
    """
    Periodic Gaussian merit function.

    Analytic expression: exp(-0.5(sin(2pi(x-epoch)/period)/s)^2)

    The introduction of the sine function breaks the traditional meaning of the standard deviation.
    So the s parameter will have to be finetuned depending on the period.

    Parameters
    ----------
    x : float
        The x value at which to evaluate the merit function.
    epoch : float
        Where the peak of the merit will be centered
    period : float
        The period of the Gaussian.
    sigma : float
        Measure of the width of each Gaussian.
    phases : list
        List of phases at which the gaussians should peak.
    """
    merit = 0.0
    for phase in phases:
        merit += np.exp(
            -0.5
            * (
                np.sin(np.pi * (observation.start_time - (epoch + phase * period)) / period)
                / sigma
            )
            ** 2
        )
    if verbose:
        print(f"current phase = {((observation.start_time - epoch) % period) / period}")
        print(f"{merit = }")
    return merit


# def moon_distance(observation: Observation, min: float = 30.0) -> float:
#     """Moon distance constraint merit function"""
# Create the AltAz frame for the moon
# TODO put this in the Night init. Only has to be done once
# moon_altaz_frame = observation.observer.moon_altaz(time=observation.time_array)
# Claculate moon distance throughout the exposure of the observation
# range_moon_distance = observation.target.coords.separation(moon_altaz_frame).deg

# TODO: Implement a merit that takes into account the moon illumination
# if range_moon_distance.min() < min:
#     return 0.0
# else:
#     return 1.0


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
