""" Definition of all the generic merit functions """

# import astropy.units as u
import numpy as np

# from astropy.time import TimeDelta
import config
from class_definitions import Observation


def time_share(
    observation: Observation, alpha: int = 3, beta: float = 5.0, delta: float = 0.2
) -> float:
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
    assert alpha > 0, "alpha for time_share merit must be greater than 0"
    assert alpha % 2 == 1, "alpha for time_share merit must be an odd positive integer"
    assert beta > 0.0, "beta for time_share merit must be greater than 0"
    assert delta > 0.0, "delta for time_share merit must be greater than 0"
    # Calculate the time share of the observation
    pct_diff = observation.target.program.time_share_pct_diff * 100
    exp_term = (pct_diff / beta) ** alpha
    abs_exp_term = abs(exp_term)
    # If the exponent is too big, cap it at 5 or -5 (after that the exp term is too big and caps
    # at the limits of 1.5 and 0.5)
    if abs_exp_term > 5:
        sign = exp_term / abs_exp_term
        exp_term = sign * 5

    # New merit
    m = (delta / (1 + np.exp(exp_term))) + (1 - delta / 2)

    # Calculate the merit
    # m = 0.5 + (1 / (1 + np.exp(exp_term)))
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
        # return 2 * (observable_range_prop - 0.5) ** 2
        return 2 * abs(observable_range_prop - 0.5)


def rise_set(observation: Observation, verbose: bool = False) -> float:
    """
    Constraint that gives higher priority to targets that are about to set for the year, or that
    are just rising for the year. This is to compensate for the culmination merit that gives priority
    to targets that are culminating at the moment. But if a target is just setting or rising, it
    doesn't culminate during the night ans so it will always have a lower priority than targets
    that culminate during the night.

    If the setting time of the target is within the first half of the night, or the rising time of
    the target is within the last half of the night, then those targets will have a higher priority.
    The cross time is calculates as the proportion of the night where either the set time or the
    rise time is. Then the merit is calculated as:

    8*(cross_time-0.5)^4 + 1

    This gives a higher merit if the target is about to set or just rising, and 1 if the target
    is culminating during the night. (approximate, I'm not checking explicitly if the target is
    culminating during the night, just if its rise or set time is within the first or last half).
    """
    # Calculate where the relevant cross time is for this target. Its to a time, but just the
    # proportion of the night where either the set time or the rise time is.
    cross_time = 0.0
    night_start = observation.night.obs_within_limits[0]
    night_end = observation.night.obs_within_limits[1]
    night_mid_point = (night_start + night_end) / 2
    if observation.set_time < night_mid_point:
        cross_time = (observation.set_time - night_start) / (night_end - night_start)
    elif observation.rise_time > night_mid_point:
        cross_time = (observation.rise_time - night_start) / (night_end - night_start)
    else:
        cross_time = 0.5

    # merit = 4 * (cross_time - 0.5) ** 2 + 1
    merit = 2 * np.abs(cross_time - 0.5) + 1
    if verbose:
        print(f"{cross_time = }")
        print(f"{merit = }")

    return merit


def culmination_mapping(observation: Observation, verbose: bool = False) -> float:
    """
    I'm tentatively calling this method "culmination mapping," though I'm still brainstorming a
    better name. It's a twist on the usual culmination merit concept. The key here is not just
    focusing on targets at their highest sky point (pure culmination), but also considering those
    that are rising or setting, which don't hit their culmination during the night.

    Here's the issue with the current approach: a setting target gets a high culmination score only
    at night's start, then it drops for the rest of the time. This means it's rarely chosen. My
    method aims to address this by shifting the focus. Instead of zeroing in on the actual physical
    culmination point, it considers all potential culmination points where a target is visible at
    night (above the altitude limit). It then aligns these with the actual observation window times.

    So, a setting target scores high at night's beginning and then declines as it sets. However,
    it's more likely to be picked because of its initial high score. A target reaching culmination
    early in the night would hit its merit peak later on. Mid-night, the true culmination syncs
    with this mapped culmination. And towards night's end, the modified merit peaks just before the
    target's actual culmination point. This means we'll be observing targets slightly after their
    culmination in the night's first half, around their peak in the middle, and a bit before their
    culmination towards the end.

    Opinion:
    I think this is still suboptimal. Because the targets that culminate early in the night will
    never be observed at actual culmination. They will always be observed a bit after. I have an
    inckling that some random process will have to be used to solved this. So when you have to
    decide between a target that is setting or a target that is culminating, its chosen randomly.
    So on average things will balance out. But this is just a hunch, I have to think about it more.
    """

    time_prop = (observation.culmination_time - observation.night.culmination_window[0]) / (
        observation.night.culmination_window[1] - observation.night.culmination_window[0]
    )
    peak_merit_time = observation.night.obs_within_limits[0] + time_prop * (
        observation.night.obs_within_limits[1] - observation.night.obs_within_limits[0]
    )

    # Calculate the merit of the target at the mapping time
    merit = gaussian((peak_merit_time - observation.start_time), 4 / 24)
    return merit


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


def gaussian(x, sigma):
    """
    A simple Gaussian.
    """
    return np.exp(-0.5 * (x / sigma) ** 2)


# def cadence(
#     observation: Observation,
#     delay: TimeDelta,
#     alpha: float,
#     beta: float = 0.5,
#     verbose: bool = False,
# ) -> float:
#     """
#     Calcualtes the merit for the desired cadence of observations.
#     It uses a piecewise function depending if the target is under or over the cadence delay.
#     If it hasn't reached the cadence yet, it uses a hyperbolic tanget function to slowly increase
#     the merit up to 0.5 when it reached the cadence. Once its over the cadence its a root function
#     that continuously increases as the observation is more overdue. The specific attack and release
#     shapes can be set with the parameters alpha and beta.
#     First the percent overdue is calculated by simply looking at how many days the target is before
#     or after the desired cadence as a proportion of the cadence. Then the exact formulas are:

#     When target is underdue:
#     0.5*(tanh(pct_overdue/alpha)+1)

#     When target is overdue:
#     0.5+(pct_overdue/50*alpha)**beta


#     Args:
#         observation (Observation): The Observation object to be used
#         delay (int): The value for the cadence in days
#         alpha (float): The attack parameter for when the target is underdue
#         beta (float): The release parameter for when the target is overdue

#     Returns:
#         score (float): The score of this merit
#     """
#     if verbose:
#         print(f"{observation.start_time = }")
#         print(f"{observation.target.last_obs = }")
#         print(f"{delay.value = }")
#     pct_overdue = (
#         observation.start_time - (observation.target.last_obs + delay.value)
#     ) / delay.value
#     if pct_overdue <= 0:
#         # Target has not reacehd the desired cadence yet
#         # Use hyperbolic tangent to gradually increase merit as it approaches the set cadence
#         return 0.5 * (np.tanh(pct_overdue / alpha) + 1)
#     else:
#         # Target has reacehd the cadence and is overdue
#         # Use a root function to slowly keep increasing the merit as the observation is increasingly overdue
#         return 0.5 + (pct_overdue / (50 * alpha)) ** beta


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


# def gaussian(x, x0, P, s):
#     """
#     Gaussian merit function.

#     Analytic expression: exp(-0.5(x/P)^2/s^2)

#     Parameters
#     ----------
#     x : float
#         The x value at which to evaluate the merit function.
#     x0 : float
#         Where the peak of the merit will be centered
#     P : float
#         The period of the Gaussian.
#     s : float
#         The standard deviation of the Gaussian.
#     """

#     return np.exp(-0.5 * (x / P) ** 2 / s**2)


# def periodic_box(x, x0, P, width):
#     """
#     Periodic box merit function.

#     It's just a box function that repeats with perio P.
#     Parameters
#     ----------
#     x : float
#         The x value at which to evaluate the merit function.
#     x0 : float
#         Where the peak of the merit will be centered
#     P : float
#         The period of the box functions.
#     width : float
#         The width of each box.
#     """

#     # TODO: Test if this is correct
#     offset = (x - x0) % P
#     return offset > -width / 2 or offset < width / 2
