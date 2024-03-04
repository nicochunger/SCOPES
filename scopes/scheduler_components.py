import inspect
import itertools
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Union

import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from astroplan import Observer
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from plotly.subplots import make_subplots


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

        # Define a middle of the night time to 4UT of the following day, which will always fall
        # within the night time range
        self.night_middle = Time(
            datetime.combine(self.night_date, datetime.min.time())
            + timedelta(days=1, hours=4)
        )
        # Get the sunset and sunrise times for the night
        self.sunset = self.observer.sun_set_time(self.night_middle, which="previous")
        self.sunrise = self.observer.sun_rise_time(self.night_middle, which="next")

        # Get the times for the different twilights
        self.civil_evening = self.observer.twilight_evening_civil(
            self.night_middle, which="previous"
        )
        self.nautical_evening = self.observer.twilight_evening_nautical(
            self.night_middle, which="previous"
        )
        self.astronomical_evening = self.observer.twilight_evening_astronomical(
            self.night_middle, which="previous"
        )
        # And the same for the morning
        self.civil_morning = self.observer.twilight_morning_civil(
            self.night_middle, which="next"
        )
        self.nautical_morning = self.observer.twilight_morning_nautical(
            self.night_middle, which="next"
        )
        self.astronomical_morning = self.observer.twilight_morning_astronomical(
            self.night_middle, which="next"
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
            self.sunset - 5 / 24, self.sunrise + 5 / 24, 300
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

        # Calculate allowed culmination times for the night
        # FOR NOW: I will calculate this very simply by saying that allowed targets are those that
        # culminate within the night time range +- 3 hours
        # TODO Change this to be more robust calcualted from the actual set of targets.
        # offsets = np.array([-3, 3]) / 24  # in days
        # self.culmination_window = self.obs_within_limits + offsets

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
            [np.any(obs.night_airmasses > 1.8) for obs in obs_list]
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


class Program:
    def __init__(
        self,
        progID: Union[int, str],
        instrument: str,
        priority: int = None,
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
        self.progID = progID
        self.instrument = instrument
        # Check that priority value is valid
        if priority < 0 or priority > 3:
            raise ValueError("Priority must be between 0 and 3")
        if not isinstance(priority, int):
            raise TypeError("Priority must be an integer")
        self.priority = priority
        # Check that plot_color is a valid hex color code
        if not (
            bool(re.search(re.compile("^#([A-Fa-f0-9]{6})$"), plot_color))
            or (plot_color is None)
        ):
            raise ValueError("plot_color must be a valid hex color code")
        self.plot_color = plot_color
        # Check that time_share_allocated is valid
        if not (0 <= time_share_allocated <= 1):
            raise ValueError("Time share must be between 0 and 1")
        self.time_share_allocated = time_share_allocated
        self.time_share_current = 0.0
        self.time_share_pct_diff = 0.0

    def update_time_share(self, current_timeshare: float):
        """
        Update the time share for the program.

        Parameters
        ----------
        current_timeshare : float
            The current time share used by the program in percent of the total
        """
        # Function that retrieves the current time used by the program
        self.time_share_current = current_timeshare
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
        priority: int = None,
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
        exposure_time : float
            The exposure time of the observation in days.
        priority : int
            The priority of the target. Must be between 0 and 3, where 0 is the highest priority
            and 3 is the lowest.
        comment : str, optional
            A comment about the target directed to the observer. Defaults to an empty string.
        """
        self.name = name
        self.program = program
        self.coords = coords
        self.ra_deg = coords.ra.deg
        self.dec_deg = coords.dec.deg
        # Check that priority value is valid
        if (priority % 1) != 0:
            raise TypeError("Priority must be an integer")
        if (priority < 0) or (priority > 3):
            raise ValueError("Priority must be between 0 and 3")
        self.priority = priority
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
        start_time : float
            The start time of the observation in JD (Julian Date).
        exposure_time : float
            The duration of the observation in days.
        night : Night
            The Night object representing the night during which the observation takes place.
        """
        self.target = target
        self.exposure_time = exposure_time
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
        self.obs_airmasses = self.night_airmasses[start_idx:end_idx]

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

    # def update_start_time(self, previous_obs):
    #     """
    #     Updates the start time of the observation based on the previous observation taking into
    #     account the overheads and instrument change.

    #     Parameters
    #     ----------
    #     previous_obs : Observation
    #         The previous observation
    #     """
    #     # TODO This entire section has to be redone to be adaptable. For now it is hardcoded
    #     # but the user should be able to say how the overheads should be calculated.
    #     sep_deg = self.separation_calc(previous_obs)
    #     # FIX: This is a dummy slew time and inst change for now, replace with fetching from config file
    #     slew_rate = 2  # degrees per second
    #     slew_time = (sep_deg / slew_rate) / 86400  # in days
    #     cor_readout = 0.0002315  # in days
    #     inst_change = 0.0
    #     if previous_obs.target.program.instrument != self.target.program.instrument:
    #         # If the instrument is different, then there is a configuration time
    #         inst_change = 0.00196759  # 170 seconds in days
    #     self.start_time = previous_obs.end_time + slew_time + cor_readout + inst_change
    #     self.end_time = self.start_time + self.exposure_time
    #     if self.end_time > self.night.obs_within_limits[1]:
    #         # Set score to 0 if observation goes beyond the end of the night
    #         self.score = 0.0
    #     else:
    #         # Update the time array becaue the start time changed
    #         self.update_alt_airmass()
    #         # Calculate new rank score based on new start time
    #         self.feasible()
    #         self.evaluate_score()

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
        # fairness = self.fairness_value
        fairness = self.fairness()

        # --- Sensibility ---
        # Veto merits that check for observatbility
        # This is calculated in the feasible method which is called every time the observation is
        # considered for the schedule
        sensibility = self.sensibility_value

        # --- Efficiency ---
        # Efficiency merits that check for scientific goal
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

    def __init__(self, slew_rate_az, slew_rate_alt):
        """
        Initialize a new instance of the Overheads class. The slew rates are given in degrees per
        second.

        Parameters
        ----------
        slew_rate_az : float
            The azimuth slew rate in degrees per second.
        slew_rate_alt : float
            The altitude slew rate in degrees per second.
        """
        self.slew_rate_az = slew_rate_az
        self.slew_rate_alt = slew_rate_alt
        self.overheads = []
        self.overhead_times = {}

    def validate_function_params(self, func):
        params = inspect.signature(func).parameters
        param_names = list(params.keys())
        expected_names = ["observation1", "observation2"]

        # Check if the function has exactly two parameters named 'observation1' and 'observation2'
        if param_names == expected_names:
            return True
        else:
            return False

    def add_overhead(self, overhead_func, can_overlap_with_slew: bool = False):
        """
        Add an overhead function to the list of overheads.

        Parameters
        ----------
        overhead_func : Callable
            The overhead function to be added.
        """
        # Check that the given object is a function
        if not inspect.isfunction(overhead_func):
            raise TypeError("overhead_func must be a function")
        # Check that the function has the correct parameters
        if not self.validate_function_params(overhead_func):
            raise ValueError(
                "Function must have exactly two parameters named 'observation1' and 'observation2'"
            )

        # Add the function to the list of overheads
        self.overheads.append((overhead_func, can_overlap_with_slew))

    def calculate_slew_time(self, obs1: Observation, obs2: Observation):
        """
        Calculate the slew time between two observations.

        Parameters
        ----------
        obs1 : Observation
            The first observation.
        obs2 : Observation
            The second observation.

        Returns
        -------
        slew_time : float
            The slew time in seconds.
        """
        # Calculate the difference in altitude and azimuth between the two observations
        time_idx = np.searchsorted(
            obs1.night.night_time_range.jd, obs1.end_time, side="right"
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

        slew_time_az = abs_sep_az / self.slew_rate_az
        slew_time_alt = abs_sep_alt / self.slew_rate_alt
        # As Az and Alt rotation happens simultaneously, total slew time will be
        # the largest of the two
        # TODO consider if alt slew time is necesarry as most of the time it will be dominated by
        # azimuth slew time (commonly larger separation than altitude)
        slew_time = np.max([slew_time_az, slew_time_alt])
        slew_time /= 86400  # Convert to days
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
        self.overhead_times["slew_time"] = slew_time

        # Calculate other overheads
        for overhead_func, can_overlap_with_slew in self.overheads:
            overhead_time = overhead_func(observation1, observation2)
            overhead_time /= 86400  # Convert to days
            self.overhead_times[overhead_func.__name__] = overhead_time
            if can_overlap_with_slew:
                total_overhead = np.max([slew_time, overhead_time])
            else:
                total_overhead = slew_time + overhead_time

        return total_overhead


class Plan:
    """
    A container class for Observation objects representing a plan for scheduling observations.
    """

    def __init__(self):
        self.observations = []
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
            observation_time = 0.0  # first_obs.exposure_time
            for obs in self.observations:
                observation_time += obs.exposure_time
            # Check that overhead and observation time add up to the total time
            overhead_time = (
                self.observations[-1].end_time - first_obs.start_time - observation_time
            )
            available_obs_time = (
                first_obs.night.obs_within_limits[1]
                - first_obs.night.obs_within_limits[0]
            )
            total_time = self.observations[-1].end_time - first_obs.start_time
            unused_time = available_obs_time - observation_time - overhead_time
            self.observation_time = TimeDelta(observation_time * u.day).to_datetime()
            self.overhead_time = TimeDelta(overhead_time * u.day).to_datetime()
            self.unused_time = TimeDelta(unused_time * u.day).to_datetime()
            self.overhead_ratio = overhead_time / total_time
            self.observation_ratio = observation_time / total_time

    def evaluate_plan(self) -> float:
        """
        Calculates the evaluation of the plan. This is the mean of the individual scores of all
        observations times the observation ratio of the plan. This is to compensate between maximum
        score of the observations but total observation time.

        Returns
        -------
        float
            The evaluation of the plan.
        """
        # Evaluate the whole observation plan
        self.score = float(
            np.mean([obs.score for obs in self.observations]) if len(self) > 0 else 0
        )  # type: ignore
        self.calculate_overhead()
        self.evaluation = self.score * self.observation_ratio
        return self.evaluation

    def print_stats(self):
        """Prints some general information and statistics of the plan."""
        self.evaluate_plan()
        self.calculate_overhead()
        print(f"Length = {len(self)}")
        print(f"Score = {self.score:.6f}")
        print(f"Evaluation = {self.evaluation:.6f}")
        print(f"Observation time = {self.observation_time}")
        print(f"Overhead time = {self.overhead_time}")
        print(f"Observation ratio = {self.observation_ratio:.5f}")
        print(f"Overhead ratio = {self.overhead_ratio:.5f}")

    def plot(self, display: bool = True, save: bool = False, path: str = None):
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

        # Plot shaded areas between sunset and civil, nautical, and astronomical evening
        y_range = np.arange(0, 91)
        ax1.fill_betweenx(
            y_range, sunset.datetime, civil_evening, color="yellow", alpha=0.2
        )
        ax1.fill_betweenx(
            y_range, civil_evening, nautical_evening, color="orange", alpha=0.2
        )
        ax1.fill_betweenx(
            y_range, nautical_evening, astronomical_evening, color="red", alpha=0.2
        )
        # Same for the morning
        ax1.fill_betweenx(
            y_range, civil_morning, sunrise.datetime, color="yellow", alpha=0.2
        )
        ax1.fill_betweenx(
            y_range, nautical_morning, civil_morning, color="orange", alpha=0.2
        )
        ax1.fill_betweenx(
            y_range, astronomical_morning, nautical_morning, color="red", alpha=0.2
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
        if save:
            plt.tight_layout()
            if path is None:
                raise ValueError("path must be specified if save is True")
            else:
                plt.savefig(path, dpi=300)
        if display:
            plt.show()
        else:
            plt.close()

    def plot_interactive(self, save: bool = False, path: str = None):
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
                + f"Exp time: {TimeDelta(obs.exposure_time * u.day).to_datetime()}<br>"
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
        if save:
            if path is None:
                raise ValueError("Path must be specified if save is True")
            else:
                fig.write_image(path, format="png")

        fig.show()

    def print_plan(self, save: bool = False, path: str = None):
        """
        Print the plan itself with each observation. This includes the target, the program,
        the start time of the observation and its exposure time.

        Parameters
        ----------
        save : bool, optional
            If True, and path is not None, save the plan to a file. Defaults to False.
        path : str, optional
            The path to the file where the plan will be saved. Defaults to None. Ignored if save is
            False.

        Returns
        -------
        str
            The night's plan as a human readable text when printed.
        """
        lines = []
        lines.append(
            f"Plan for the night of {self.observations[0].night.night_date} (Times in UTC)"
        )
        lines.append("--------------------------------------------------\n")
        # lines.append(" #      Program ID      Target                  Start time   (Exp time)")
        lines.append(
            f"{'#':<6}{'Program ID':<12}{'Target':<20}{'Start time':<13}{'(Exp time)':<12}{'Comment for the observer':<40}"
        )
        for i, obs in enumerate(self.observations):
            start_time = Time(obs.start_time, format="jd").datetime
            exp_time = TimeDelta(obs.exposure_time * u.day).to_datetime()
            string = f"{f'{i+1:>2}:':<6}"
            string += (
                f"{f'{obs.target.program.progID} {obs.target.program.instrument}':<12}"
            )
            string += f"{obs.target.name:<20}"
            string += f"{start_time.strftime('%H:%M:%S'):<13}"
            string += f"{f'({exp_time})':<12}"
            string += f"{obs.target.comment:<40}"
            lines.append(string)
        plan_txt = "\n".join(lines)
        if save:
            if path is None:
                raise ValueError("Path must be specified if save is True")
            else:
                with open(path, "w") as f:
                    f.write(plan_txt)

        return plan_txt

    def __len__(self):
        return len(self.observations)

    def __str__(self):
        return self.print_plan(save=False)
