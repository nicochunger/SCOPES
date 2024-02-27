""" Definition of all the generic merit functions """

import numpy as np
from astropy.time import Time

from .scheduler_components import Observation


def time_share(
    observation: Observation, alpha: int = 3, beta: float = 5.0, delta: float = 0.1
) -> float:
    """
    Time share fairness merit. It uses a modified sigmoid function to calculate the merit.
    The specific shape can be set with the parameters alpha and beta. The exact formula is:

    m = (delta / (1 + np.exp((pct_diff / beta) ** alpha))) + (1 - delta / 2)

    It's shaped in a way so that there is some permissiveness. This means that a program
    can be over or under the allocated time by a certain percentage before its priority is decreased
    or increased. The alpha parameter controls how sharp the sigmoid is, and the beta parameter
    controls how much difference is allowed (in percentage).

    Parameters
    ----------
    observation : Observation
        The Observation object to be used
    alpha : float, optional
        The attack parameter for how sharp the sigmoid is. Defaults to 3.
    beta : float, optional
        The leeway parameter for how much difference in time use is allowed. Defaults to 5.
    delta : float, optional
        The maximum percentage increase or decrease that will be applied if a program is over or
        under its allocated time. Defaults to 0.1.

    Returns
    -------
    float : The time share merit of the observation
    """
    # Check that the parameters are valid
    if alpha <= 0:
        raise ValueError("alpha for time_share merit must be greater than 0")
    if alpha % 2 == 0:
        raise ValueError("alpha for time_share merit must be an odd positive integer")
    if beta <= 0.0:
        raise ValueError("beta for time_share merit must be greater than 0")
    if delta <= 0.0:
        raise ValueError("delta for time_share merit must be greater than 0")

    # Calculate the time share of the observation
    pct_diff = observation.target.program.time_share_pct_diff * 100
    exp_term = (pct_diff / beta) ** alpha

    # If the exponent term is too big, cap it at 5 or -5
    # This is to limit the size of the exponent term and control the np.exp() function.
    # After 5 the merit already reaches its limits
    abs_exp_term = abs(exp_term)
    if abs_exp_term > 5:
        sign = exp_term / abs_exp_term
        exp_term = sign * 5

    # merit
    m = (delta / (1 + np.exp(exp_term))) + (1 - delta / 2)
    return m


def priority(
    observation: Observation,
    prog_base: float = 1.0,
    prog_offset: float = 0.1,
    tar_base: float = 0.0,
    tar_offset: float = 0.05,
) -> float:
    """
    Priority merit function. This is a fairness merit function to schedule based on priority.
    In this implementation it assumes that the priority values given is between 0 and 3, where
    0 is the highest priority and 3 is the lowest. This priority value is mapped to a base values
    for the priority level of 2, and then an offset is added above or below that base value depending
    if the priority is 3, 1, or 0. This is to convert this priority scale to something close to
    0 or 1 usually.

    The default values reflect a system where the program priorities are centered at 1, and then
    the target priorities are centered at 0 which work as a modifier to the program priority. This
    means that the priority of the program takes precedent, and then the priority of the targets
    help to decide within the program which targets to observe first. There is a small overlap,
    where a target with priority 0 can have a mapped priority higher than a target with priority 3
    of the next highest priority program. At this stage we don't have to worry that this generates
    more observations for the higher priority programs because that is balanced by the time share
    merit.

    Parameters
    ----------
    observation : Observation
        The Observation object to be used
    prog_base : float
        The base value for the priority level of 2, for programs.
    prog_offset : float
        The offset to be added to the base value of programs for the priority levels of 3, 1 and 0.
    tar_base : float
        The base value for the priority level of 2, for targets.
    tar_offset : float
        The offset to be added to the base value of targets for the priority levels of 3, 1 and 0.
    """
    # Validate priority value for program and target exists
    if observation.target.program.priority is None:
        raise ValueError("Program priority value is None")
    if observation.target.priority is None:
        raise ValueError("Target priority value is None")

    map_prog_priority = prog_base + prog_offset * (
        2 - observation.target.program.priority
    )
    map_tar_priority = tar_base + tar_offset * (2 - observation.target.priority)
    return map_prog_priority + map_tar_priority


def at_night(observation: Observation) -> float:
    """
    Merit function that returns 1 if the observation is within the chosen night time limits, and
    0 otherwise.

    Parameters
    ----------
    observation : Observation
        The Observation object to be used
    """
    return float(
        observation.start_time >= observation.night.obs_within_limits[0]
        and observation.end_time <= observation.night.obs_within_limits[1]
    )


def airmass(observation: Observation, max: float, alpha: float = 0.05) -> float:
    """
    Merit function on the airmass of the observation. It uses a hyperbolic tangent function to
    gradually increase the merit as the airmass increases. The specific shape can be set with the
    parameters min and alpha. The exact formula is:

    m = tanh((max - current_airmass) / alpha)

    Parameters
    ----------
    observation : Observation
        The Observation object to be used
    max : float, optional
        The maximum airmass allowed for this observation to be considered.
    alpha : float, optional
        A measure of the tolerance around the maximum airmass. Defaults to 0.05.
    """
    # Claculate airmass throughout the exposure of the observation
    if len(observation.obs_airmasses) == 0:
        return 0.0
    else:
        return observation.obs_airmasses.max() < max

    # Same but with the tangent function
    # if len(observation.obs_airmasses) == 0:
    #     return 0.0
    # else:
    #     arg = (max - observation.obs_airmasses.max()) / alpha
    #     return np.tanh(arg)


def altitude(
    observation: Observation,
    min: float = 0,
    max: float = 90,
) -> float:
    """
    Altitude constraint merit function

    Parameters
    ----------
    observation : Observation
        The Observation object to be used
    """
    # Claculate altitude throughout the exposure of the observation
    if len(observation.obs_altitudes) == 0:
        return 0.0
    else:
        return (observation.obs_altitudes.min() > min) and (
            observation.obs_altitudes.max() < max
        )


def culmination(observation: Observation, verbose: bool = False) -> float:
    """
    Culmination constraint merit function.
    This merit calculates the current height of the target and the proportion to the maximum height
    it will reach during the current night.
    """
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


def culmination_mapping(observation: Observation, verbose: bool = False) -> float:
    """
    ObservationEfficiency

    This merit is designed to make the scheduling of astronomical observations more efficient by
    expanding the selection of observable stars beyond those reaching their culmination (highest
    point in the sky) during the night. The normal Culmination merit misses out on stars that
    culminate before the night begins or after it ends, even though these stars are still
    observable at acceptable altitudes during the night.

    To address this, it uses an extended time range for calculating merits beyond the actual
    observable hours of the night. This extended range includes the earliest culmination point of
    any star still observable at the night's start and the latest culmination point of any star
    still observable at the night's end.

    This extended timerange is then mapped to the actual night time range where observations will
    be taken (typically within nautical or astronomical twilights). Its a simple one-to-one mapping
    between two different time ranges. Then to calcualte the merit, the time at which the star
    actually culminates is mapped from the larger timerange to the actual night range. That new
    mapped time is when that star will peak in this merit function.

    The method shifts observation priorities throughout the night—prioritizing setting stars early
    on, stars near their culmination around the middle, and rising stars towards the end. This
    method is a compromise between observing stars at their highest point but also giving priority
    to stars that are setting or rising.


    Parameters
    ----------
    observation : Observation
        The Observation object to be used
    verbose : bool, optional
        If True, print the calculated merit. Defaults to False.
    """

    time_prop = (
        observation.culmination_time - observation.night.culmination_window[0]
    ) / (
        observation.night.culmination_window[1]
        - observation.night.culmination_window[0]
    )
    peak_merit_time = observation.night.obs_within_limits[0] + time_prop * (
        observation.night.obs_within_limits[1] - observation.night.obs_within_limits[0]
    )

    # Calculate the merit of the target at the mapping time
    merit = gaussian((peak_merit_time - observation.start_time), 4 / 24)

    if verbose:
        print(f"{time_prop = }")
        print(f"{peak_merit_time = }")
        print(f"{merit = }")

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
    phases : list, optional
        List of phases at which the gaussians should peak. Defaults to [0.0].
    verbose : bool, optional
        If True, print the calculated merit. Defaults to False.
    """
    merit = 0.0
    for phase in phases:
        merit += np.exp(
            -0.5
            * (
                np.sin(
                    np.pi * (observation.start_time - (epoch + phase * period)) / period
                )
                / sigma
            )
            ** 2
        )
    if verbose:
        print(f"current phase = {((observation.start_time - epoch) % period) / period}")
        print(f"{merit = }")
    return merit


def time_critical(
    observation: Observation,
    start_time: float,
    start_time_tolerance: float,
    steepness: float = 0.0014,  # in days, which is ~2 minutes
    verbose: bool = False,
) -> float:
    """
    Calculate the time criticality merit of an observation. It uses a double hyperbolic tangent
    function to gradually increase the merit as the observation approaches the desired start time.
    The center times of the increasing tanh and decresing tanh, are start_time - start_time_tolerance
    and start_time + start_time_tolerance, respectively.

    TODO Rethink this merit and how to ensure that time critical observations are done without
    failure. This merit is not enough to ensure that the observation is done at the desired time
    as a previous observation can cover the entire tolerance range and thus block the time critical
    observation from being done.

    Parameters
    ----------
    observation : Observation
        The observation object.
    start_time : float
        The desired start time for the observation. In Julian Date (JD).
    start_time_tolerance : float
        The tolerance around the desired start time in days.
    steepness : float, optional
        The steepness of the hyperbolic tangent function. A measure of how much time it takes the
        function to go from 0 to the max value in days. Defaults to 0.0014 days, which is ~2 minutes.
    verbose : bool, optional
        If True, print the calculated merit. Defaults to False.

    Returns:
    --------
    float: The time criticality merit of the observation.
    """
    arg1 = (observation.start_time - (start_time - start_time_tolerance)) / steepness
    arg2 = ((start_time + start_time_tolerance) - observation.start_time) / steepness
    merit = np.tanh(arg1) + np.tanh(arg2)

    if verbose:
        print(f"time_critical {merit = }")
    return merit


def moon_distance(observation: Observation, min: float = 30.0) -> float:
    """
    Moon distance constraint merit function
    TODO: this entire merit function has to be tested.

    Parameters
    ----------
    observation : Observation
        The Observation object to be used
    min : float, optional
        The minimum distance to the moon in degrees. Defaults to 30.0.
    """
    # Create the AltAz frame for the moon
    # TODO put this in the Night init.
    # Can be done only once if calculated beforehand for the entire night
    moon_altaz_frame = observation.night.observer.moon_altaz(
        time=Time(observation.start_time, format="jd")
    )
    # Calculate moon distance throughout the exposure of the observation
    range_moon_distance = observation.target.coords.separation(moon_altaz_frame).deg

    # TODO: Implement a merit that takes into account the moon illumination and distance
    if range_moon_distance.min() < min:
        return 0.0
    else:
        return 1.0


def gaussian(x, sigma):
    """
    A simple Gaussian.

    Parameters
    ----------
    x : float
        The x value at which to evaluate the Gaussian.
    sigma : float
        Measure of the width of the Gaussian.
    """
    return np.exp(-0.5 * (x / sigma) ** 2)
