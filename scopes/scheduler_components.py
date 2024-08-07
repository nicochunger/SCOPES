import inspect
import itertools
import re
import uuid
import warnings
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from astroplan import Observer
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from plotly.subplots import make_subplots
from timezonefinder import TimezoneFinder


class Night:
    def __init__(self, night_date: date, observations_within: str, observer: Observer):
        """
        Initialize a new instance of the Night class.

        Parameters
        ----------
        night_date : date
            The date of the night.
        observations_within : {"civil", "nautical", "astronomical"}
            Within which twilight observations will be done. Can be one of "civil", "nautical",
            or "astronomical".
        observer : astroplan.Observer
            An astroplan.Observer object that defines where the telescope is located in the world.
        """
        self.night_date = night_date
        self.observations_within = observations_within
        # Check if the observations_within is valid
        valid_options = ["civil", "nautical", "astronomical"]
        if self.observations_within not in valid_options:
            raise ValueError(f"observations_within must be one of {valid_options}")
        self.observer = observer

        # Calculate the solar midnight for the night based on the observer's location
        self.solar_midnight = self.calculate_solar_midnight()
        # Get the sunset and sunrise times for the night
        self.sunset = self.observer.sun_set_time(self.solar_midnight, which="previous")
        self.sunrise = self.observer.sun_rise_time(self.solar_midnight, which="next")

        # Get the times for the different twilights
        self.civil_evening = self.observer.twilight_evening_civil(
            self.solar_midnight, which="previous"
        )
        self.nautical_evening = self.observer.twilight_evening_nautical(
            self.solar_midnight, which="previous"
        )
        self.astronomical_evening = self.observer.twilight_evening_astronomical(
            self.solar_midnight, which="previous"
        )
        # And the same for the morning
        self.civil_morning = self.observer.twilight_morning_civil(
            self.solar_midnight, which="next"
        )
        self.nautical_morning = self.observer.twilight_morning_nautical(
            self.solar_midnight, which="next"
        )
        self.astronomical_morning = self.observer.twilight_morning_astronomical(
            self.solar_midnight, which="next"
        )

        # Time ranges for the different twilights
        self.time_range_solar = np.linspace(self.sunset, self.sunrise, 300)
        self.time_range_civil = np.linspace(self.civil_evening, self.civil_morning, 300)
        self.time_range_nautical = np.linspace(
            self.nautical_evening, self.nautical_morning, 300
        )
        self.time_range_astronomical = np.linspace(
            self.astronomical_evening, self.astronomical_morning, 300
        )
        # Extended time range for the night, 5 hours before and after sunset / sunrise
        self.time_range_extended = np.linspace(
            self.sunset - TimeDelta(5 / 24, format="jd"),
            self.sunrise + TimeDelta(5 / 24, format="jd"),
            300,
        )
        # Now only use the jd value of the twilights
        self.civil_evening = self.civil_evening.jd
        self.nautical_evening = self.nautical_evening.jd
        self.astronomical_evening = self.astronomical_evening.jd
        self.astronomical_morning = self.astronomical_morning.jd
        self.nautical_morning = self.nautical_morning.jd
        self.civil_morning = self.civil_morning.jd

        # Define the night time range based on the chosen twilight as the start and end times
        # of the observable night
        if self.observations_within == "civil":
            self.obs_within_limits = np.array([self.civil_evening, self.civil_morning])
            self.night_time_range = self.time_range_civil
        elif self.observations_within == "nautical":
            self.obs_within_limits = np.array(
                [self.nautical_evening, self.nautical_morning]
            )
            self.night_time_range = self.time_range_nautical
        elif self.observations_within == "astronomical":
            self.obs_within_limits = np.array(
                [self.astronomical_evening, self.astronomical_morning]
            )
            self.night_time_range = self.time_range_astronomical

        # Save the duration of the night
        self.night_duration = (
            self.night_time_range[-1] - self.night_time_range[0]
        ).to_datetime()

        # Calculate the alt-az frame of the moon for the night
        self.moon_altaz_frame = self.observer.moon_altaz(self.night_time_range)

    def calculate_solar_midnight(self):
        """
        Calculate the solar midnight for the night.

        Returns
        -------
        solar_midnight : astropy.time.Time
            The calculated solar midnight in UTC.
        """
        # Use timezonefinder to get the timezone
        timezone_str = TimezoneFinder().timezone_at(
            lat=self.observer.latitude.value, lng=self.observer.longitude.value
        )

        # Define the observer's local timezone
        local_timezone = pytz.timezone(timezone_str)

        # Create a localized datetime object
        local_datetime = datetime.combine(self.night_date, time(hour=18))
        local_datetime = local_timezone.localize(local_datetime)

        # Convert local datetime to UTC
        utc_datetime = local_datetime.astimezone(pytz.utc)

        # Convert the UTC datetime to an astropy Time object
        time_utc = Time(utc_datetime)

        # Calculate solar midnight in UTC based on the converted UTC time
        solar_midnight = self.observer.midnight(time_utc, which="next")
        return solar_midnight

    def calculate_culmination_window(self, obs_list):
        """
        Calculate the culmination window for the night based on the given list of observations.

        Parameters
        ----------
        obs_list : List[Observation]
            A list of Observation objects.
        """

        # Check if the objects in obs_list have the attribute culmination_time
        if not all(hasattr(obs, "culmination_time") for obs in obs_list):
            raise AttributeError(
                "sky_path() has to be run for all observations before calling this method."
            )

        # Get the culmination times of all observations
        culm_times = np.array([obs.culmination_time for obs in obs_list])
        # Check the observation is observable during the night
        is_observable = np.array(
            [np.any(obs.night_airmasses < 1.8) for obs in obs_list]
        )

        # Calculate the start and end times of the culmination window
        culm_window_start = np.min(culm_times[is_observable])
        culm_window_end = np.max(culm_times[is_observable])
        self.culmination_window = (culm_window_start, culm_window_end)
        # return culmination_window

    def __str__(self):
        lines = [
            f"Night(Date: {self.night_date},",
            f"      Sunset: {self.sunset},",
            f"      Sunrise: {self.sunrise},",
            f"      Civil evening: {self.civil_evening},",
            f"      Nautical evening: {self.nautical_evening},",
            f"      Astronomical evening: {self.astronomical_evening},",
            f"      Civil morning: {self.civil_morning},",
            f"      Nautical morning: {self.nautical_morning},",
            f"      Astronomical morning: {self.astronomical_morning},",
            f"      Observations within: '{self.observations_within}')",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Night):
            return False
        else:
            return self.night_date == other.night_date


class Instrument:
    def __init__(self, name: str, instrument_type: str = "", plot_color: str = None):
        """
        Initialize a new instance of the Instrument class.

        Parameters
        ----------
        name : str
            The name of the instrument.
        instrument_type : str, optional
            The type of the instrument. Default is an empty string: "".
        plot_color : str, optional
            The color to use when plotting an observation of this instrument. Must be a valid hex
            code. By default the colors will be chosen from the 'Pastel1' color pallette from
            matplotlib.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.name = name
        if not isinstance(instrument_type, str):
            raise TypeError("instrument_type must be a string")
        self.instrument_type = instrument_type
        # Check that plot_color is a valid hex color code if its not None
        if plot_color is not None:
            if not bool(re.search(re.compile("^#([A-Fa-f0-9]{6})$"), plot_color)):
                raise ValueError(
                    "plot_color must be a valid hex color code, e.g. '#FF0000'"
                )
        self.plot_color = plot_color

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        lines = [
            "Instrument(",
            f"    Name = {self.name}",
        ]
        if self.instrument_type:
            lines.append(f"    Type = {self.instrument_type}")
        if self.plot_color:
            lines.append(f"    Plot color = {self.plot_color}")
        return "\n".join(lines)

    def __eq__(self, other):
        if not isinstance(other, Instrument):
            return False
        else:
            return self.name == other.name


class Program:
    def __init__(
        self,
        progID: str,
        instrument: Instrument,
        priority: int,
        time_share_allocated: float = 0.0,
        plot_color: str = None,
    ):
        """
        Initialize a new instance of the Program class.

        Parameters
        ----------
        progID : int or str
            The program ID code.
        instrument : str
            The name of the instrument that the program uses.
        time_share_allocated : float within [0, 1]
            The time share allocated to the program as a percentage of total time.
            Must be between 0 and 1.
        priority : int
            The priority of the program. Must be between 0 and 3, where 0 is the highest priority
            and 3 is the lowest.
        priority_base : int, optional
            The base value for mapping the priority. Defaults to 1.
        priority_offset : float, optional
            The offset value for mapping the priority. Defaults to 0.1.
        plot_color : str, optional
            The color to use when plotting an observation of this program. Must be a valid hex code.
            By default the colors will be chosen from the 'Set2' color pallette from matplotlib.
        """
        if not isinstance(progID, str):
            raise TypeError("progID must be a string")
        self.progID = progID
        if not isinstance(instrument, Instrument):
            raise TypeError("instrument must be of type Instrument")
        self.instrument = instrument
        # Check that priority value is valid
        if priority < 0 or priority > 3:
            raise ValueError("Priority must be between 0 and 3")
        if not isinstance(priority, int):
            raise TypeError("Priority must be an integer")
        self.priority = priority
        # Check that plot_color is a valid hex color code if its not None
        if plot_color is not None:
            if not (bool(re.search(re.compile("^#([A-Fa-f0-9]{6})$"), plot_color))):
                raise ValueError(
                    "plot_color must be a valid hex color code, e.g. '#FF0000'"
                )
        self.plot_color = plot_color
        # Check that time_share_allocated is valid
        if not (0 <= time_share_allocated <= 1):
            raise ValueError("Time share must be between 0 and 1")
        self.time_share_allocated = time_share_allocated
        # Initialize the current time usage to the allocated time share
        self.set_current_time_usage(self.time_share_allocated)

    def set_current_time_usage(self, current_time_usage: float):
        """
        Update the time share for the program.

        Parameters
        ----------
        current_time_usage : float
            The current time share used by the program in percent of the total
        """
        # Value check
        if not (0 <= current_time_usage <= 1):
            raise ValueError("current_time_usage must be between 0 and 1")
        # Update the current time share and calculate the difference
        self.time_share_current = current_time_usage
        self.time_share_pct_diff = self.time_share_current - self.time_share_allocated

    def __str__(self) -> str:
        lines = [
            "Program(",
            f"    ID = {self.progID}",
            f"    Instrument = {self.instrument}",
            f"    Time allocated = {self.time_share_allocated}",
            f"    Priority = {self.priority})",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class Merit:
    def __init__(
        self,
        name: str,
        func: Callable,
        merit_type: str,
        parameters: Dict[str, Any] = {},
    ):
        """
        Initialize a new instance of the Merit class.

        Parameters
        ----------
        name : str
            The name of the merit.
        func : Callable
            The function that computes the merit.
        merit_type : {"fairness", "veto", "efficiency"}
            The type of the merit. Can be one of "fairness", "veto", or "efficiency".
        parameters : Dict[str, Any], optional
            Custom parameters for the merit function. Defaults to {}. The keys of the dictionary
            must match the names of the parameters of the function.
        """
        self.name = name
        self.func = func  # The function that computes this merit
        self.description = self.func.__doc__
        self.merit_type = merit_type  # "veto" or "efficiency"
        self.parameters = parameters  # Custom parameters for this merit

        # Check that the merit type is valid
        if self.merit_type not in ["fairness", "veto", "efficiency"]:
            raise ValueError(
                f"Invalid merit type ({self.merit_type}). "
                "Valid types are 'fairness', 'veto' and 'efficiency'."
            )

        # Consistency checks between the given func and parameters
        # It checks that the required parameters are all there, and that there are no extra
        # parameters that are not part of the function
        required_func_parameters = []
        optional_func_parameters = []
        for name, param in inspect.signature(self.func).parameters.items():
            if param.default == inspect.Parameter.empty:
                required_func_parameters.append(name)
            else:
                optional_func_parameters.append(name)
        # Check that the first parameter is "observation"
        if not (required_func_parameters[0] == "observation"):
            raise KeyError("The first parameter has to be 'observation'")
        # Check that the given parameters match the required parameters of the function
        if not set(required_func_parameters[1:]).issubset(set(self.parameters.keys())):
            raise ValueError(
                f"The given parameters ({set(self.parameters.keys())}) don't match the "
                "required parameters of the given function "
                f"({set(required_func_parameters[1:])})"
            )
        # Check that there are no extra parameters that are not part of the function
        if not set(self.parameters.keys()).issubset(
            set(required_func_parameters + optional_func_parameters)
        ):
            raise KeyError(
                "There are given parameters that are not part of the given function"
            )

    def evaluate(self, observation, **kwargs) -> float:
        """
        Evaluate the function with the given observation and additional arguments.

        Parameters
        ----------
        observation : Observation
            The input observation.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float: The evaluation result.
        """
        # Combine custom parameters and runtime arguments, then call the function
        all_args = {**self.parameters, **kwargs}
        return self.func(observation, **all_args)

    def __str__(self):
        return f"Merit({self.name}, {self.merit_type}, {self.parameters})"

    def __repr__(self):
        return self.__str__()


class Target:
    def __init__(
        self,
        name: str,
        program: Program,
        coords: SkyCoord,
        priority: Union[int, None] = None,
        comment: str = "",
    ):
        """
        Initialize a new instance of the Target class.

        Parameters
        ----------
        name : str
            The name of the target.
        prog : Program
            The Program object that the target belongs to.
        coords : SkyCoord
            The coordinates of the target.
        priority : int
            The priority of the target. Must be between 0 and 3, where 0 is the highest priority
            and 3 is the lowest.
        comment : str, optional
            A comment about the target directed to the observer. Defaults to an empty string.
        """
        # Value checks
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.name = name
        if not isinstance(program, Program):
            raise TypeError("program must be of type Program")
        self.program = program
        if not isinstance(coords, SkyCoord):
            raise TypeError("coords must be of type SkyCoord")
        self.coords = coords
        self.ra_deg = coords.ra.deg
        self.dec_deg = coords.dec.deg
        # Check that priority value is valid
        if priority is not None:
            if not isinstance(priority, int):
                raise TypeError(
                    f"Priority must be an integer, given: {priority} ({type(priority)})"
                )
        if isinstance(priority, int):
            if (priority < 0) or (priority > 3):
                raise ValueError(f"Priority must be between 0 and 3, given: {priority}")
        self.priority = priority
        if not isinstance(comment, str):
            raise TypeError(f"comment must be a string, given: '{comment}'")
        self.comment = comment

        self.fairness_merits: List[Merit] = []  # List of all fairness merits
        self.efficiency_merits: List[Merit] = []  # List of all efficiency merits
        self.veto_merits: List[Merit] = []  # List to store veto merits

    def add_merit(self, merit: Merit):
        """
        Adds a merit to the corresponding list based on its merit type.

        Parameters
        ----------
        merit : Merit
            The merit object to be added
        """
        if not isinstance(merit, Merit):
            raise TypeError("merit must be of type Merit")
        if merit.merit_type == "fairness":
            self.fairness_merits.append(merit)
        elif merit.merit_type == "veto":
            self.veto_merits.append(merit)
        elif merit.merit_type == "efficiency":
            self.efficiency_merits.append(merit)

    def add_merits(self, merits: List[Merit]):
        """
        Adds a list of Merit objects to the instance.

        Parameters
        ----------
        merits : List[Merit]
            A list of Merit objects to be added
        """
        if not isinstance(merits, list):
            raise TypeError("merits must be a list")
        if not all(isinstance(merit, Merit) for merit in merits):
            raise TypeError("the objects in merits must be of type Merit")
        for merit in merits:
            self.add_merit(merit)

    def __str__(self):
        lines = [
            f"Target(Name: {self.name},",
            f"       Program: {self.program.progID},",
            f"       Coordinates: ({self.coords.ra.deg:.3f}, {self.coords.dec.deg:.3f}),",
            f"       Priority: {self.priority},",
            f"       Fairness Merits: {self.fairness_merits},",
            f"       Veto Merits: {self.veto_merits},",
            f"       Efficiency Merits: {self.efficiency_merits})",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Target):
            return False
        else:
            return self.name == other.name


class Observation:
    def __init__(
        self,
        target: Target,
        exposure_time: float,
    ):
        """
        Initialize a new instance of the Observation class.

        Parameters
        ----------
        target : Target
            The Target object representing the target being observed.
        exposure_time : float
            The exposure time, or duration, of the observation in seconds.
        """
        if not isinstance(target, Target):
            raise TypeError("target must be of type Target")
        self.target = target
        # Check if exposure time makes sense
        if exposure_time <= 0:
            raise ValueError("exposure_time must be greater than 0")
        # Check if exposure time is less than 1 second
        if exposure_time < 1:
            warnings.warn("exposure_time is less than 1 second")
        # Check if exposure time is greater than 9 hours
        if exposure_time > 32400:
            warnings.warn("exposure_time is greater than 9 hours")
        self.exposure_time = exposure_time / 86400  # Convert to days for internal use
        # Telescope altitude limits. These are only used to calculate the minimum and maximum
        # altitudes of the target during the night. The actual observational limits are set by the
        # Altitude merit defined by the user.
        self.tel_alt_lower_lim = 10.0
        self.tel_alt_upper_lim = 90.0
        self.score: float = 0.0  # Initialize score to zero
        self.veto_merits: List[float] = []  # List to store veto merits
        self.unique_id = uuid.uuid4()  # Unique ID for the observation instance

    def set_night(self, night: Night):
        """
        Set the night for the observation.

        Parameters
        ----------
        night : Night
            The Night object representing the night during which the observation takes place.
        """
        self.night = night

    def set_start_time(self, start_time: float):
        """
        Set the start time of the observation.

        Parameters
        ----------
        start_time : float
            The start time of the observation in JD (Julian Date).
        """
        # Check night has been assigned
        if not hasattr(self, "night"):
            raise AttributeError("Night must be assigned to the observation first")
        # Check if the start time is within the night time range
        if not (
            self.night.night_time_range[0].jd
            <= start_time
            <= self.night.night_time_range[-1].jd
        ):
            raise ValueError("Start time must be within the night time range")

        self.start_time = start_time
        self.end_time = self.start_time + self.exposure_time

    def skypath(self):
        """
        Calculate the skypath of the target during the night.
        To be run at the start of the scheduling process.
        """
        # Create the AltAz frame for the observation during the night
        self.night_altaz_frame = self.target.coords.transform_to(
            self.night.observer.altaz(time=self.night.night_time_range)
        )

        # Get the altitudes and airmasses of the target during the night
        self.night_altitudes = self.night_altaz_frame.alt.deg
        self.night_azimuths = self.night_altaz_frame.az.deg
        self.night_airmasses = self.night_altaz_frame.secz
        # Update the altitudes and airmasses for the observation timerange
        self.update_alt_airmass()
        # Get the minimum altitude of the target during the night
        self.min_altitude = max(self.night_altitudes.min(), self.tel_alt_lower_lim)
        # Get the maximum altitude of the target during the night
        self.max_altitude = min(self.night_altitudes.max(), self.tel_alt_upper_lim)

        # Convert extended time range to AltAz frame and get the time of maximum altitude
        self.culmination_time = self.night.time_range_extended[
            np.argmax(
                self.target.coords.transform_to(
                    self.night.observer.altaz(time=self.night.time_range_extended)
                ).alt.deg
            )
        ].jd

        # Get the rise and set times of the target
        start_time_astropy = self.night.night_time_range[0]
        if self.night_altitudes[0] > self.tel_alt_lower_lim:
            # If the target is already up by night start, the rise time is "previous"
            self.rise_time = self.night.observer.target_rise_time(
                start_time_astropy,
                self.target.coords,
                horizon=self.tel_alt_lower_lim * u.deg,
                which="previous",
                n_grid_points=10,
            ).jd
            # and set time is "next"
            self.set_time = self.night.observer.target_set_time(
                start_time_astropy,
                self.target.coords,
                horizon=self.tel_alt_lower_lim * u.deg,
                which="next",
                n_grid_points=10,
            ).jd
        else:
            # If the target is not up by night start, the rise time is "next"
            self.rise_time = self.night.observer.target_rise_time(
                start_time_astropy,
                self.target.coords,
                horizon=self.tel_alt_lower_lim * u.deg,
                which="next",
                n_grid_points=10,
            ).jd
            # and set time is also "next"
            self.set_time = self.night.observer.target_set_time(
                start_time_astropy,
                self.target.coords,
                horizon=self.tel_alt_lower_lim * u.deg,
                which="next",
                n_grid_points=10,
            ).jd

    def fairness(self) -> float:
        """
        Calculate the fairness score of the target.

        The fairness score is calculated by multiplying the priority of the target+program
        with the product of the evaluations of all fairness merits associated with the target.

        Returns
        -------
        float
            The fairness score of the target.
        """
        if len(self.target.fairness_merits) == 0:
            return 1.0
        else:
            return np.prod(
                [merit.evaluate(self) for merit in self.target.fairness_merits]
            )

    def update_alt_airmass(self):
        """
        Update the altitude and airmass values throughout the observation based on the start time
        and exposure time of the observation.
        """
        # Find indices
        night_range_jd = self.night.night_time_range.value
        start_idx = np.searchsorted(night_range_jd, self.start_time, side="left")
        end_idx = np.searchsorted(night_range_jd, self.end_time, side="right")
        self.obs_time_range = night_range_jd[start_idx:end_idx]
        self.obs_altitudes = self.night_altitudes[start_idx:end_idx]
        self.obs_azimuths = self.night_azimuths[start_idx:end_idx]
        self.obs_airmasses = self.night_airmasses[start_idx:end_idx].value

    def feasible(self, verbose: bool = False) -> float:
        """
        Determines the feasibility of the target based on the veto merits.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the name and value of each veto merit. Defaults to False

        Returns
        -------
        float
            The sensibility value, which is the product of all veto merit values.
        """
        veto_merit_values = []
        for merit in self.target.veto_merits:
            value = merit.evaluate(self)
            veto_merit_values.append(value)
            if verbose:
                print(f"{merit.name}: {value}")
            if value == 0.0:
                break
        self.sensibility_value = np.prod(veto_merit_values)
        return self.sensibility_value  # type: ignore

    def evaluate_score(self, verbose: bool = False) -> float:
        """
        Evaluates the score of the observation based on fairness, sensibility, and efficiency.

        Parameters
        ----------
        verbose : bool, optional
            If True, print the fairness, sensibility, efficiency, and rank score.

        Returns
        -------
        float
            The score of the observation.
        """
        # --- Fairness ---
        # Balances time allocation and priority
        fairness = self.fairness()

        # --- Sensibility ---
        # Veto merits that check for observatbility
        # This is calculated in the feasible method which is called every time the observation is
        # considered for the schedule
        sensibility = self.sensibility_value

        # --- Efficiency ---
        # Efficiency merits that check for scientific goal
        if len(self.target.efficiency_merits) == 0:
            efficiency = 1.0
        else:
            efficiency = np.mean(
                [merit.evaluate(self) for merit in self.target.efficiency_merits]
            )

        # --- Rank Score ---
        # Calculate total rank score by taking the product of fairness, sensibility and efficiency
        self.score = np.prod([fairness, sensibility, efficiency])  # type: ignore

        # Print results if verbose
        if verbose:
            print(f"Fairness: {fairness}")
            print(f"Sensibility: {sensibility}")
            print(f"Efficiency: {efficiency}")
            print(f"Rank score: {self.score}")

        return self.score

    def update_start_and_score(self, start_time):
        """
        Update the start time of the observation and recalculate the score.

        Parameters
        ----------
        start_time : float
            The new start time to set for the observation
        """
        self.set_start_time(start_time)
        self.update_alt_airmass()
        self.feasible()
        self.evaluate_score()

    def __str__(self):
        lines = [
            f"Observation(Target: {self.target.name},",
            f"            Start time: {self.start_time},",
            f"            Exposure time: {self.exposure_time},",
            f"            Score: {self.score})",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        else:
            return self.unique_id == other.unique_id


class Overheads:
    """
    A class for managing the calculation of the overheads between two consecutive observations.

    This class handles the calculation of various overheads involved in transitioning from one
    observation position to another.
    """

    def __init__(
        self,
        slew_rate_az: float,
        slew_rate_alt: float,
        cable_wrap_angle: Optional[float] = None,
    ):
        """
        Initialize a new instance of the Overheads class. The slew rates are given in degrees per
        second.

        Parameters
        ----------
        slew_rate_az : float
            The azimuth slew rate in degrees per second.
        slew_rate_alt : float
            The altitude slew rate in degrees per second.
        cable_wrap_angle : float, optional
            The azimuth angle where the cable wrap limit is at, in degrees. Default is None.
        """
        # Validate that slew rates are positive
        if slew_rate_az <= 0 or slew_rate_alt <= 0:
            raise ValueError("Slew rates must be positive")
        # Validate that cable wrap angle is between 0 and 360 degrees
        if cable_wrap_angle is not None:
            if not (0 <= cable_wrap_angle <= 360):
                raise ValueError("Cable wrap angle must be between 0 and 360 degrees")

        self.slew_rate_az = slew_rate_az
        self.slew_rate_alt = slew_rate_alt
        self.cable_wrap_angle = cable_wrap_angle
        self.overheads = []

    def _validate_function_params(self, func):
        """
        Validate the parameters of the given function.

        Parameters
        ----------
        func : Callable
            The function to validate.

        Returns
        -------
        bool
            True if the function has exactly two parameters named 'observation1' and 'observation2',
            False otherwise.
        """
        params = inspect.signature(func).parameters
        param_names = list(params.keys())
        expected_names = ["observation1", "observation2"]

        # Check if the function has exactly two parameters named 'observation1' and 'observation2'
        if param_names == expected_names:
            return True
        else:
            return False

    def _is_angle_between(
        self, start_angle: float, end_angle: float, check_angle: float
    ) -> bool:
        """
        Check if an angle is between two other angles.

        Parameters
        ----------
        start_angle : float
            The start angle in degrees.
        end_angle : float
            The end angle in degrees.
        check_angle : float
            The angle to check if it is between the start and end angles.

        Returns
        -------
        bool
            True if the check angle is between the start and end angles, False otherwise.
        """
        # Normalize angles to ensure they are between 0 and 360 degrees
        start_angle %= 360
        end_angle %= 360
        check_angle %= 360

        # Find the normalized positions of end and check angles relative to the start angle
        end_relative = (end_angle - start_angle) % 360
        check_relative = (check_angle - start_angle) % 360

        # Check if the check_angle is between start_angle and end_angle on the shortest path
        return (
            check_relative <= end_relative
            if end_relative <= 180
            else check_relative >= end_relative
            or check_relative <= (end_relative - 360)
        )

    def add_overhead(self, overhead_func, can_overlap_with_slew: bool = False):
        """
        Add an overhead function to the list of overheads.

        The function must have exactly two parameters named 'observation1' and 'observation2'.

        Parameters
        ----------
        overhead_func : Callable
            The overhead function to be added.
        """
        # Check that the given object is a function
        if not inspect.isfunction(overhead_func):
            raise TypeError("overhead_func must be a function")
        # Check that the function has the correct parameters
        if not self._validate_function_params(overhead_func):
            raise ValueError(
                "Function must have exactly two parameters named 'observation1' and 'observation2'"
            )

        # Add the function to the list of overheads
        self.overheads.append((overhead_func, can_overlap_with_slew))

    def calculate_slew_time(self, obs1: Observation, obs2: Observation):
        """
        Calculate the slew time between two observations, taking into account cable wrap.

        Parameters
        ----------
        obs1 : Observation
            The first observation.
        obs2 : Observation
            The second observation.
        cable_wrap_angle : float
            The azimuth angle where the cable wrap limit is at, in degrees.

        Returns
        -------
        slew_time : float
            The slew time in seconds.
        """
        # Calculate the difference in altitude and azimuth between the two observations
        time_idx = np.searchsorted(
            obs1.night.night_time_range.jd, obs1.end_time, side="left"
        )
        obs1_alt = obs1.night_altitudes[time_idx]
        obs1_az = obs1.night_azimuths[time_idx]
        obs2_alt = obs2.night_altitudes[time_idx]
        obs2_az = obs2.night_azimuths[time_idx]

        abs_sep_alt = np.abs(obs1_alt - obs2_alt)
        abs_sep_az = np.abs(obs1_az - obs2_az)
        # Take shortest path if azimuth separation is larger than 180 degrees
        if abs_sep_az > 180:
            abs_sep_az = 360 - abs_sep_az

        # Check if cable wrap is between the two azimuths when taking the shortest path
        if self.cable_wrap_angle is not None:
            if self._is_angle_between(obs1_az, obs2_az, self.cable_wrap_angle):
                # If cable wrap is between the two azimuths, azimuth slew has to go through long way
                abs_sep_az = 360 - abs_sep_az

        slew_time_az = abs_sep_az / self.slew_rate_az
        slew_time_alt = abs_sep_alt / self.slew_rate_alt
        # As Az and Alt slew happens simultaneously, total slew time will be the largest of the two
        slew_time = max([slew_time_az, slew_time_alt]) / 86400  # Convert to days
        return slew_time

    def calculate_transition(self, observation1, observation2):
        """
        Calculate the total overhead time between two observations.

        Parameters
        ----------
        observation1 : Observation
            The first observation.
        observation2 : Observation
            The second observation.
        """
        # Calculate slew time based on RA and DEC differences and slew rates
        slew_time = self.calculate_slew_time(observation1, observation2)

        total_overhead = slew_time
        extra_overhead_time = 0.0

        # Calculate other overheads
        for overhead_func, can_overlap_with_slew in self.overheads:
            overhead_time = (
                overhead_func(observation1, observation2) / 86400
            )  # Convert to days

            if can_overlap_with_slew:
                total_overhead = np.max([total_overhead, overhead_time])
            else:
                extra_overhead_time += overhead_time

        total_overhead += extra_overhead_time

        return total_overhead

    def __str__(self) -> str:
        return f"Overheads(\n\tslew_rate_az={self.slew_rate_az}, \n\tslew_rate_alt={self.slew_rate_alt}, \n\tcable_wrap_limit={self.cable_wrap_angle})"


class Plan:
    """
    A container class for Observation objects representing a plan for scheduling observations.
    """

    def __init__(self):
        self.observations: List[Observation] = []
        self.score = 0.0
        self.evaluation = 0.0

    def add_observation(self, observation: Observation):
        """
        Add an observation to the plan.

        Parameters
        ----------
        observation : Observation
            The Observation object to be added to the plan.
        """
        self.observations.append(observation)
        return self

    def calculate_overhead(self):
        """
        Calculates the overheads for the entire plan, as well as the total observation time.
        """
        # Calculate the overheads for the plan
        # Go through all observation and count the time between the end of one observation and the
        # start of the next one
        if len(self) == 0:
            self.observation_time = 0.0
            self.overhead_time = 0.0
            self.unused_time = 0.0
            self.overhead_ratio = 0.0
            self.observation_ratio = 0.0
            return
        else:
            first_obs = self.observations[0]
            observation_time = np.sum([obs.exposure_time for obs in self.observations])
            # Check that overhead and observation time add up to the total time
            total_time = self.observations[-1].end_time - first_obs.start_time
            overhead_time = total_time - observation_time
            available_obs_time = (
                first_obs.night.obs_within_limits[1]
                - first_obs.night.obs_within_limits[0]
            )
            unused_time = available_obs_time - observation_time - overhead_time

            self.observation_time = timedelta(days=observation_time)
            self.overhead_time = timedelta(days=overhead_time)
            self.unused_time = timedelta(days=unused_time)
            self.overhead_ratio = overhead_time / total_time
            self.observation_ratio = observation_time / total_time

    def calculate_avg_airmass(self):
        """
        Calculate the average airmass of the observations in the plan.
        """
        self.avg_airmass = np.mean(
            [
                np.max(obs.obs_airmasses)
                for obs in self.observations
                if len(obs.obs_airmasses) > 0
            ]
        )

    def evaluate_plan(self, w_score: float = 0.2, w_overhead: float = 0.8) -> float:
        """
        Calculates the evaluation of the plan. This is the mean of the individual scores of all
        observations times the observation ratio of the plan. This is to compensate between maximum
        score of the observations but total observation time.

        Parameters
        ----------
        w_score : float, optional
            The weight of the score in the evaluation. Defaults to 0.4.
        w_overhead : float, optional
            The weight of the overhead in the evaluation. Defaults to 0.6.

        Returns
        -------
        float
            The evaluation of the plan.
        """
        if w_score + w_overhead != 1:
            raise ValueError("The weights must sum to 1")
        # Evaluate the whole observation plan
        self.score = float(
            np.mean([obs.score for obs in self.observations]) if len(self) > 0 else 0
        )  # type: ignore
        # Calculate the overheads for the plan
        self.calculate_overhead()
        # Calculate the average airmass of the observations in the plan
        self.calculate_avg_airmass()
        # Calculate the evaluation of the plan
        # self.evaluation = self.score * self.observation_ratio
        self.evaluation = w_score * self.score + w_overhead * self.observation_ratio
        return self.evaluation

    def print_stats(self):
        """Prints some general information and statistics of the plan."""
        self.evaluate_plan()
        print(f"Length = {len(self)}")
        print(f"Score = {self.score:.6f}")
        print(f"Evaluation = {self.evaluation:.6f}")
        print(f"Observation time = {self.observation_time}")
        print(f"Overhead time = {self.overhead_time}")
        print(f"Observation ratio = {self.observation_ratio:.5f}")
        print(f"Overhead ratio = {self.overhead_ratio:.5f}")
        print(f"Avg airmass = {self.avg_airmass:.5f}")

    def plot(self, display: bool = True, path: str = None):
        """
        Plot the schedule for the night.

        Parameters
        ----------
        display : bool, optional
            Option to display the plot. Defaults to True.
        save : bool, optional
            If True, and path is given, save the plot to a file. Defaults to False.
        path : str, optional
            The path to the file where the plot will be saved. Defaults to None. Ignored if save is
            False.
        """
        first_obs = self.observations[0]

        # Get sunset and sunrise times for this night
        night = first_obs.night
        sunset = Time(night.sunset, format="jd")
        sunrise = Time(night.sunrise, format="jd")

        # Get the times for the different twilights
        civil_evening = Time(night.civil_evening, format="jd").datetime
        nautical_evening = Time(night.nautical_evening, format="jd").datetime
        astronomical_evening = Time(night.astronomical_evening, format="jd").datetime
        civil_morning = Time(night.civil_morning, format="jd").datetime
        nautical_morning = Time(night.nautical_morning, format="jd").datetime
        astronomical_morning = Time(night.astronomical_morning, format="jd").datetime

        # Get which programs are part of this plan
        programs = list(
            set([obs.target.program for obs in self.observations])
        )  # Remove duplicates
        # Define unique colors for each program
        # if the programs have their plot_color attribute set, use that color
        # otherwise use the default color ('Set2' color pallette from matplotlib)
        default_colors = itertools.cycle(
            [mcolors.rgb2hex(color) for color in plt.get_cmap("Set2").colors]
        )
        prog_colors = {}
        for prog in programs:
            if prog.plot_color is None:
                prog_colors[prog.progID] = next(default_colors)
            else:
                prog_colors[prog.progID] = prog.plot_color

        fig, ax1 = plt.subplots(figsize=(13, 5))

        # Plot the altitude tracks of the targets
        for i, obs in enumerate(self.observations):
            # TODO clean up this part by using existing variables in the obs objects
            solar_altaz_frame = obs.target.coords.transform_to(
                night.observer.altaz(time=night.time_range_solar)
            )
            solar_night_altitudes = solar_altaz_frame.alt.deg

            # Plot altitude tracks of the target
            # Through the entire night
            ax1.plot_date(
                Time(obs.night.time_range_solar, format="jd").datetime,
                solar_night_altitudes,
                "-.",
                c="gray",
                alpha=0.6,
                lw=0.3,
            )
            # Only the observed period in highlighted color
            ax1.plot_date(
                Time(obs.obs_time_range, format="jd").datetime,
                obs.obs_altitudes,
                "-",
                c=prog_colors[obs.target.program.progID],
                lw=2,
                solid_capstyle="round",
            )

        # TODO Plot the tracks of the moon

        # Plot shaded areas between sunset and civil, nautical, and astronomical evening
        y_range = np.arange(0, 91)
        alpha_tw_fill = 0.2
        ax1.fill_betweenx(
            y_range, sunset.datetime, civil_evening, color="yellow", alpha=alpha_tw_fill
        )
        ax1.fill_betweenx(
            y_range,
            civil_evening,
            nautical_evening,
            color="orange",
            alpha=alpha_tw_fill,
        )
        ax1.fill_betweenx(
            y_range,
            nautical_evening,
            astronomical_evening,
            color="red",
            alpha=alpha_tw_fill,
        )
        # Same for the morning
        ax1.fill_betweenx(
            y_range,
            civil_morning,
            sunrise.datetime,
            color="yellow",
            alpha=alpha_tw_fill,
        )
        ax1.fill_betweenx(
            y_range,
            nautical_morning,
            civil_morning,
            color="orange",
            alpha=alpha_tw_fill,
        )
        ax1.fill_betweenx(
            y_range,
            astronomical_morning,
            nautical_morning,
            color="red",
            alpha=alpha_tw_fill,
        )
        # Add text that have the words "civil", "nautical", and "astronomical".
        # These boxes are placed vertically at the times of each of them (both evening and morning)
        text_kwargs = {
            "rotation": 90,
            "verticalalignment": "bottom",
            "color": "gray",
            "fontsize": 8,
        }

        ax1.text(
            sunset.datetime, 30.5, "Sunset", horizontalalignment="right", **text_kwargs
        )
        ax1.text(
            civil_evening, 30.5, "Civil", horizontalalignment="right", **text_kwargs
        )
        ax1.text(
            nautical_evening,
            30.5,
            "Nautical",
            horizontalalignment="right",
            **text_kwargs,
        )
        ax1.text(
            astronomical_evening,
            30.5,
            "Astronomical",
            horizontalalignment="right",
            **text_kwargs,
        )
        ax1.text(
            (civil_morning + timedelta(minutes=3)),
            30.5,
            "Civil",
            horizontalalignment="left",
            **text_kwargs,
        )
        ax1.text(
            (nautical_morning + timedelta(minutes=3)),
            30.5,
            "Nautical",
            horizontalalignment="left",
            **text_kwargs,
        )
        ax1.text(
            (astronomical_morning + timedelta(minutes=3)),
            30.5,
            "Astronomical",
            horizontalalignment="left",
            **text_kwargs,
        )
        ax1.text(
            (sunrise.datetime + timedelta(minutes=3)),
            30.5,
            "Sunrise",
            horizontalalignment="left",
            **text_kwargs,
        )

        # Use DateFormatter to format x-axis to only show time
        time_format = mdates.DateFormatter("%H:%M")
        ax1.xaxis.set_major_formatter(time_format)

        # Set the major ticks to an hourly interval
        hour_locator = mdates.HourLocator(interval=1)
        ax1.xaxis.set_major_locator(hour_locator)

        # Add a legend at the bottom of the plot to identiy the program colors
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color=prog_colors[prog.progID],
                lw=2,
                label=f"{prog.progID}",
            )
            for prog in programs
        ]
        ax1.legend(handles=legend_elements, loc="lower center", ncol=len(programs))

        # In the title put the date of the schedule
        plt.title(f"Schedule for the night of {self.observations[0].night.night_date}")
        ax1.set_xlabel("Time [UTC]")
        ax1.set_ylabel("Altitude [deg]")
        ax1.set_ylim(30, 90)

        # Add a second axis to show the airmass
        # Set up airmass values and compute the corresponding altitudes for those airmass values
        desired_airmasses = np.arange(1.8, 0.9, -0.1)
        corresponding_altitudes = list(
            90.0 - np.degrees(np.arccos(1.0 / desired_airmasses[:-1]))
        )
        corresponding_altitudes.append(90.0)

        # Create the secondary y-axis for Airmass
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())
        # Set y-ticks at computed altitudes for desired airmasses
        ax2.set_yticks(corresponding_altitudes)
        ax2.set_yticklabels(
            np.round(desired_airmasses, 2)
        )  # Display the desired airmass values
        ax2.set_ylabel("Airmass")
        ax2.tick_params("y")
        if path is not None:
            plt.tight_layout()
            plt.savefig(path, dpi=300)
        if display:
            plt.show()
        else:
            plt.close()

    def plot_interactive(self, path: str = None):
        """
        Makes an interactive plot of the Plan using Plotly.

        Parameters
        ----------
        save : bool, optional
            If True, and path is given, save the plot to a file. Defaults to False.
        path : str, optional
            The path to the file where the plot will be saved. Defaults to None. Ignored if save is
            False.
        """
        # Import Plotly here to avoid unnecessary dependencies
        import plotly.graph_objs as go

        first_obs = self.observations[0]

        # Get sunset and sunrise times for this night
        night = first_obs.night
        sunset = Time(night.sunset, format="jd")
        sunrise = Time(night.sunrise, format="jd")

        # Get the times for the different twilights
        civil_evening = Time(night.civil_evening, format="jd").datetime
        nautical_evening = Time(night.nautical_evening, format="jd").datetime
        astronomical_evening = Time(night.astronomical_evening, format="jd").datetime
        civil_morning = Time(night.civil_morning, format="jd").datetime
        nautical_morning = Time(night.nautical_morning, format="jd").datetime
        astronomical_morning = Time(night.astronomical_morning, format="jd").datetime

        # Get which programs are part of this plan
        programs = list(
            set([obs.target.program for obs in self.observations])
        )  # Remove duplicates
        # Define unique colors for each program
        # if the programs have their plot_color attribute set, use that color
        # otherwise use the default color ('Set2' color pallette from matplotlib)
        default_colors_iter = itertools.cycle(
            [mcolors.rgb2hex(color) for color in plt.get_cmap("Set2").colors]
        )

        # Create a dictionary that maps each program to a color
        prog_colors = {}
        for prog in programs:
            if prog.plot_color is None:
                prog_colors[prog.progID] = next(default_colors_iter)
            else:
                prog_colors[prog.progID] = prog.plot_color

        # Create subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Track which programs have been added to the legend
        added_to_legend = set()

        # Plot each observation
        for i, obs in enumerate(self.observations):
            # Time range and altitude calculations
            # TODO clean up this part by using existing variables in the obs objects
            time_range = Time(
                np.linspace(obs.start_time, (obs.start_time + obs.exposure_time), 20),
                format="jd",
            ).datetime
            altitudes = obs.target.coords.transform_to(
                night.observer.altaz(time=time_range)
            ).alt.deg
            # Generate an array of equally spaced Julian Dates
            num_points = 300  # The number of points you want
            jd_array = np.linspace(sunset.jd, sunrise.jd, num_points)

            # Convert back to Astropy Time objects
            night_time_array = Time(jd_array, format="jd", scale="utc").datetime

            night_altitudes = obs.target.coords.transform_to(
                night.observer.altaz(time=night_time_array)
            ).alt.deg

            # Determine whether to add to legend
            program_id = obs.target.program.progID
            add_to_legend = program_id not in added_to_legend
            if add_to_legend:
                added_to_legend.add(program_id)

            # Hover text
            hovertemplate = (
                f"<b>{obs.target.name}</b><br>"
                + f"{obs.target.program.instrument} {program_id}<br><br>"
                + f"Start time: {Time(obs.start_time, format='jd').datetime.strftime('%H:%M:%S %d-%m-%Y')}<br>"
                + f"Exp time: {timedelta(obs.exposure_time)}<br>"
                + f"Comment: {obs.target.comment}"
                + "<extra></extra>"
            )

            # Plotting the target
            fig.add_trace(
                go.Scatter(
                    x=night_time_array,
                    y=night_altitudes,
                    mode="lines",
                    line=dict(color="gray", dash="dot", width=0.3),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=time_range,
                    y=altitudes,
                    mode="lines",
                    line=dict(color=prog_colors[program_id], width=2),
                    name=f"{program_id}",
                    text=f"{obs.target.name}",
                    hovertemplate=hovertemplate,
                    # hovertext=hover_text,
                    # hoverinfo="text",
                    showlegend=add_to_legend,
                ),
                secondary_y=False,
            )

        # Twilight zones plotting
        shaded_dicts = {}
        shades = list(
            zip(
                [
                    sunset.datetime,
                    civil_evening,
                    nautical_evening,
                    astronomical_evening,
                    astronomical_morning,
                    nautical_morning,
                    civil_morning,
                    sunrise.datetime,
                ],
                ["", "yellow", "orange", "red", "", "red", "orange", "yellow"],
            )
        )
        # print(shades)
        for i, pair in enumerate(shades):
            if (i == 0) or (i == 4):
                continue
            twilight, color = pair
            shaded_dicts[twilight] = dict(
                type="rect",
                x0=shades[i - 1][0],
                y0=0,
                x1=twilight,
                y1=90,
                line=dict(width=0),
                fillcolor=color,
                opacity=0.2,
            )
        for twilight in shaded_dicts:
            fig.add_shape(**shaded_dicts[twilight])

        # Add text annotations for twilight times
        # fig.add_annotation(
        #     x=sunset.datetime, y=30.5, text="Sunset", showarrow=False, xanchor="right"
        # )
        # Repeat for other twilight times

        # Set up airmass values and compute the corresponding altitudes for those airmass values
        desired_airmasses = np.arange(1.8, 0.9, -0.1)
        corresponding_altitudes = list(
            90.0 - np.degrees(np.arccos(1.0 / desired_airmasses[:-1]))
        )
        corresponding_altitudes.append(90.0)

        # Formatting x-axis and y-axis
        fig.update_xaxes(title_text="Time [UTC]", tickformat="%H:%M", tickangle=45)
        fig.update_yaxes(title_text="Altitude [deg]", range=[30, 90], secondary_y=False)
        fig.update_yaxes(
            title_text="Airmass",
            tickvals=corresponding_altitudes,
            ticktext=np.round(desired_airmasses, 2),
            secondary_y=True,
        )

        # Legend and title
        fig.update_layout(
            title=f"Schedule for the night of {self.observations[0].night.night_date}",
            legend_title_text="Program IDs",
            legend=dict(orientation="v", xanchor="left", yanchor="top"),
            plot_bgcolor="white",
        )

        # Saving the plot if requested
        if path is not None:
            fig.write_image(path, format="png")

        fig.show()

    def plot_polar(self, display: bool = True, path: str = None):
        # Create polar plot
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        for i, obs in enumerate(self.observations):
            # Plot each observation as a line in the polar plot
            ax.plot(
                np.radians(obs.obs_azimuths),
                obs.obs_altitudes,
                # label=obs.target.program.progID,
                color=obs.target.program.plot_color,
            )
            if i == 0:
                # Plot a greend point at the start of the night
                ax.plot(
                    np.radians(obs.obs_azimuths[0]),
                    obs.obs_altitudes[0],
                    "go",
                    label="Start of the plan",
                )
            elif i == len(self.observations) - 1:
                # Plot a red point at the end of the night
                ax.plot(
                    np.radians(obs.obs_azimuths[-1]),
                    obs.obs_altitudes[-1],
                    "ro",
                    label="End of the plan",
                )

            # Connect the last point of the current observation with the first point of the next observation
            if i < len(self.observations) - 1:
                ax.plot(
                    [
                        np.radians(obs.obs_azimuths[-1]),
                        np.radians(self.observations[i + 1].obs_azimuths[0]),
                    ],
                    [obs.obs_altitudes[-1], self.observations[i + 1].obs_altitudes[0]],
                    ls="dashed",
                    lw=0.6,
                    c="gray",
                )

        # Additional plot formatting
        ax.set_theta_zero_location("N")  # Set 0 degrees at the top
        ax.set_theta_direction(-1)  # Set the rotation of the plot clockwise
        ax.set_title("Azimuth vs. Altitude", va="bottom")
        ax.set_ylim(90, 30)
        plt.legend(loc="best")

        # Saving the plot if requested
        if path is not None:
            plt.tight_layout()
            plt.savefig(path, format="png")

        if display:
            plt.show()
        else:
            plt.close()

    def _instruments_analysis(self):
        """
        Analyze the instruments used in the plan. It registers which instruments are used in this
        plan and assigns a default plot color if the user didn't provide one.
        """
        self.instruments: List[Instrument] = []
        for obs in self.observations:
            instrument = obs.target.program.instrument
            if instrument not in self.instruments:
                self.instruments.append(instrument)

        # Assign default colors to instruments that don't have a color assigned
        default_colors = itertools.cycle(
            [mcolors.rgb2hex(color) for color in plt.get_cmap("Pastel1").colors]
        )
        for instrument in self.instruments:
            if instrument.plot_color is None:
                instrument.plot_color = next(default_colors)

    def plot_altaz(self, display: bool = True, path: str = None) -> None:
        """
        Plot the azimuth and altitude of the observations.

        Parameters
        ----------
        display : bool, optional
            Option to display the plot. Defaults to True.
        save : bool, optional
            If True, and path is given, save the plot to a file. Defaults to False.
        path : str, optional
            The path to the file where the plot will be saved. Defaults to None. Ignored if save is
            False.
        """
        # Initialize the figure and subplots
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Analyse the instruments in this plan
        self._instruments_analysis()

        # Iterate through observations and plot them
        for i, obs in enumerate(self.observations):
            # Create arrays of JD for each observation point
            jd_times = np.linspace(obs.start_time, obs.end_time, len(obs.obs_azimuths))

            # TODO if a line crosses the 0/360 deg boundary create a split right at the boundary
            # and plot the two parts separately. This is so that the line is not drawn across the
            # entire plot when switching sides.
            # Plot azimuths and altitudes
            ax1.plot(jd_times, obs.obs_azimuths, "-", lw=1, c="k")
            ax2.plot(jd_times, obs.obs_altitudes, "-", lw=1, c="k")

            # Plot diagonal line between this obs and start of the next
            if i < len(self.observations) - 1:
                ax1.plot(
                    [jd_times[-1], self.observations[i + 1].start_time],
                    [obs.obs_azimuths[-1], self.observations[i + 1].obs_azimuths[0]],
                    "-"
                    if abs(
                        obs.obs_azimuths[-1] - self.observations[i + 1].obs_azimuths[0]
                    )
                    <= 180
                    else "--",
                    lw=1,
                    c="k",
                )
                ax2.plot(
                    [jd_times[-1], self.observations[i + 1].start_time],
                    [obs.obs_altitudes[-1], self.observations[i + 1].obs_altitudes[0]],
                    "-",
                    lw=1,
                    c="k",
                )

        # Shaded regions
        for i in range(len(self.observations)):
            obs = self.observations[i]
            # Shade the observation time color coded by instrument
            ax1.axvspan(
                obs.start_time,
                obs.end_time,
                color=obs.target.program.instrument.plot_color,
                alpha=0.7,
            )
            ax2.axvspan(
                obs.start_time,
                obs.end_time,
                color=obs.target.program.instrument.plot_color,
                alpha=0.7,
            )
            if i < len(self.observations) - 1:
                # Shade the overhead between observations
                end_current_obs = obs.end_time
                start_next_obs = self.observations[i + 1].start_time
                if start_next_obs > end_current_obs:  # Ensure there is a gap
                    ax1.axvspan(
                        end_current_obs, start_next_obs, color="grey", alpha=0.8
                    )
                    ax2.axvspan(
                        end_current_obs, start_next_obs, color="grey", alpha=0.8
                    )

        # Use scatter plot for the legend markers
        for inst in self.instruments:
            ax1.scatter(
                [],
                [],
                label=inst.name,
                color=inst.plot_color,
                edgecolor="none",
                s=100,
                marker="s",
            )

        # Adjusting legend
        ax1.legend(loc="best", title="Instruments")

        # Set the labels and titles
        ax1.set_ylabel("Azimuth Angle [deg]")
        ax1.set_ylim(0, 360)
        ax2.set_ylabel("Elevation Angle [deg]")
        ax2.set_ylim(30, 90)
        ax2.set_xlabel("Observation Time [JD]")

        # Set titles and layout
        ax1.set_title(
            "Azimuth\n(dashed line means it actually crosses 0/360 deg and goes the other way than shown)"
        )
        ax2.set_title("Elevation")

        # Saving the plot if requested
        if path is not None:
            plt.tight_layout()
            plt.savefig(path, format="png")

        if display:
            plt.show()
        else:
            plt.close()

    def to_df(self):
        """
        Create a pandas DataFrame of the plan and return it. It saves this table in an attribute
        called 'df'.
        """
        # List to hold the rows for the DataFrame
        data = []

        def format_timedelta(tdelta):
            """Format timedelta to display only hours, minutes, and seconds."""
            total_seconds = int(tdelta.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        for obs in self.observations:
            start_time = Time(obs.start_time, format="jd").datetime
            exp_time = timedelta(days=obs.exposure_time)
            ra_str = obs.target.coords.ra.to_string(
                unit="hour", sep=":", precision=0, pad=True
            )
            dec_str = obs.target.coords.dec.to_string(
                unit="deg", sep=":", precision=0, pad=True, alwayssign=True
            )

            # Add a dictionary for each observation
            data.append(
                {
                    "Instrument": obs.target.program.instrument,
                    "ProgID": obs.target.program.progID,
                    "Target": obs.target.name,
                    "RA": ra_str,
                    "DEC": dec_str,
                    "Start Time": start_time.strftime("%H:%M:%S"),
                    "Exp Time": format_timedelta(exp_time),
                    "Exp Time (s)": int(exp_time.total_seconds()),
                    "Comment": obs.target.comment,
                }
            )

        # Create the DataFrame from the list of dictionaries
        columns = [
            "Instrument",
            "ProgID",
            "Target",
            "RA",
            "DEC",
            "Start Time",
            "Exp Time",
            "Exp Time (s)",
            "Comment",
        ]
        self.df = pd.DataFrame(data, columns=columns)

        return self.df

    def to_csv(self, *args, **kwargs):
        """
        Create a CSV file for the pandas DataFrame representation of the Plan
        This method can take any keyword argument that the pd.DataFrame.to_csv() method can take.
        """
        if not hasattr(self, "df"):
            self.to_df()
        self.df.to_csv(*args, **kwargs)

    def __len__(self):
        return len(self.observations)

    def __repr__(self):
        return f"<Plan> with {len(self)} observations"

    def __str__(self):
        return self.to_df().to_string()
