import inspect
import itertools
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List

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

import config

# Define global observer location
lasilla = Observer.at_site("lasilla")


class Night:
    def __init__(self, night_date: date, observations_within: str, observer: Observer):
        # How to calculate which night to take from the input time?
        # For now just take the night of the input time
        self.night_date = night_date
        self.observer = observer
        self.midnight = Time(
            datetime.combine(self.night_date, datetime.min.time()) + timedelta(days=1, hours=4)
        )
        self.sunset = self.observer.sun_set_time(self.midnight, which="previous")
        self.sunrise = self.observer.sun_rise_time(self.midnight, which="next")

        # Get the times for the different twilights
        self.civil_evening = self.observer.twilight_evening_civil(self.midnight, which="previous")
        self.nautical_evening = self.observer.twilight_evening_nautical(
            self.midnight, which="previous"
        )
        self.astronomical_evening = self.observer.twilight_evening_astronomical(
            self.midnight, which="previous"
        )
        # And the same for the morning
        self.civil_morning = self.observer.twilight_morning_civil(self.midnight, which="next")
        self.nautical_morning = self.observer.twilight_morning_nautical(
            self.midnight, which="next"
        )
        self.astronomical_morning = self.observer.twilight_morning_astronomical(
            self.midnight, which="next"
        )
        # Time ranges for the different twilights
        self.time_range_solar = np.linspace(self.sunset, self.sunrise, 300)
        self.time_range_civil = np.linspace(self.civil_evening, self.civil_morning, 300)
        self.time_range_nautical = np.linspace(self.nautical_evening, self.nautical_morning, 300)
        self.time_range_astronomical = np.linspace(
            self.astronomical_evening, self.astronomical_morning, 300
        )
        # Now only use the jd value of the twilights
        self.civil_evening = self.civil_evening.jd
        self.nautical_evening = self.nautical_evening.jd
        self.astronomical_evening = self.astronomical_evening.jd
        self.civil_morning = self.civil_morning.jd
        self.nautical_morning = self.nautical_morning.jd
        self.astronomical_morning = self.astronomical_morning.jd

        self.observations_within = observations_within
        # Check if the observations_within is valid
        valid_options = ["civil", "nautical", "astronomical"]
        if self.observations_within not in valid_options:
            raise ValueError(f"observations_within must be one of {valid_options}")
        if self.observations_within == "civil":
            self.obs_within_limits = np.array([self.civil_evening, self.civil_morning])
            self.night_time_range = self.time_range_civil
        elif self.observations_within == "nautical":
            self.obs_within_limits = np.array([self.nautical_evening, self.nautical_morning])
            self.night_time_range = self.time_range_nautical
        elif self.observations_within == "astronomical":
            self.obs_within_limits = np.array(
                [self.astronomical_evening, self.astronomical_morning]
            )
            self.night_time_range = self.time_range_astronomical

        # Calculate allowed culmination times for the night
        # FOR NOW: I will calculate this very simply by saying that allowed targets are those that
        # culminate within the night time range +- 3 hours
        offsets = np.array([-3, 3]) / 24  # in days
        self.culmination_window = self.obs_within_limits + offsets

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
        self, progID: int, instrument: str, time_share_allocated: float, plot_color: str = None
    ):
        self.progID = progID
        self.instrument = instrument
        self.plot_color = plot_color
        assert bool(re.search(re.compile("^#([A-Fa-f0-9]{6})$"), self.plot_color)) or (
            self.plot_color is None
        ), "plot_color must be a valid hex color code"
        self.time_share_allocated = time_share_allocated
        assert (self.time_share_allocated >= 0.0) and (
            self.time_share_allocated <= 1.0
        ), "Time share must be between 0 and 1"
        self.time_share_current = 0.0
        self.time_share_pct_diff = 0.0

    def update_time_share(self, current_timeshare: float):
        # Function that retrieves the current time used by the program
        self.time_share_current = current_timeshare
        self.time_share_pct_diff = (
            self.time_share_current - self.time_share_allocated
        ) / self.time_share_allocated

    def __str__(self) -> str:
        print(f"Program({self.progID}, {self.instrument}, {self.time_share_allocated})")


class Merit:
    def __init__(
        self,
        name: str,
        func: Callable,
        merit_type: str,
        parameters: Dict[str, Any] = {},
    ):
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
        # paramters that are not part of the function
        required_func_parameters = []
        optional_func_parameters = []
        for name, param in inspect.signature(self.func).parameters.items():
            if param.default == inspect.Parameter.empty:
                required_func_parameters.append(name)
            else:
                optional_func_parameters.append(name)
        assert (
            required_func_parameters[0] == "observation"
        ), "The first parameter has to be 'observation'"
        assert set(required_func_parameters[1:]).issubset(set(self.parameters.keys())), (
            f"The given parameters ({set(self.parameters.keys())}) don't match the "
            "required parameters of the given function "
            f"({set(required_func_parameters[1:])})"
        )
        assert set(self.parameters.keys()).issubset(
            set(required_func_parameters + optional_func_parameters)
        ), "There are given parameters that are not part of the given function"

    def evaluate(self, observation, **kwargs) -> float:
        # Combine custom parameters and runtime arguments, then call the function
        all_args = {**self.parameters, **kwargs}
        return self.func(observation, **all_args)

    def __str__(self):
        return f"Merit({self.name}, {self.merit_type}, {self.parameters})"

    def __repr__(self):
        return self.__str__()


class Target:
    def __init__(
        self, name: str, prog: Program, coords: SkyCoord, priority: int, exposure_time: float
    ):
        self.name = name
        self.program = prog
        self.coords = coords
        self.ra_deg = coords.ra.deg
        self.dec_deg = coords.dec.deg
        self.exposure_time = exposure_time
        # self.last_obs = last_obs  # in Julian Date
        self.priority = self.renormalize_priority(priority)
        self.fairness_merits: List[Merit] = []  # List of all fairness merits
        self.efficiency_merits: List[Merit] = []  # List of all efficiency merits
        self.veto_merits: List[Merit] = []  # List to store veto merits

    def add_merits(self, merits: List[Merit]):
        """Add a list of merits to the target"""
        assert isinstance(merits, list), "merits must be a list of Merit objects"
        for merit in merits:
            self.add_merit(merit)

    def add_merit(self, merit: Merit):
        """Add a merit to the target"""
        if merit.merit_type == "fairness":
            self.fairness_merits.append(merit)
        elif merit.merit_type == "veto":
            self.veto_merits.append(merit)
        elif merit.merit_type == "efficiency":
            self.efficiency_merits.append(merit)

    def renormalize_priority(self, priority: int):
        """Renormalize the priority of the target"""
        return (4 - priority) / 4

    def __str__(self):
        lines = [
            f"Target(Name: {self.name},",
            f"       Program: {self.program.progID},",
            f"       Coordinates: ({self.coords.ra.deg:.3f}, {self.coords.dec.deg:.3f}),",
            # f"       Last observation: {self.last_obs},",
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
        start_time: float,
        exposure_time: float,
        night: Night,
    ):
        self.target = target
        self.start_time = start_time
        self.exposure_time = exposure_time
        self.end_time = self.start_time + self.exposure_time
        self.night = night
        self.score: float = 0.0  # Initialize score to zero
        self.veto_merits: List[float] = []  # List to store veto merits
        self.unique_id = uuid.uuid4()
        self.fairness_value = self.fairness()

        # Create the AltAz frame for the observation during the night
        self.night_altaz_frame = self.target.coords.transform_to(
            self.night.observer.altaz(time=self.night.night_time_range)
        )

        # Get the altitudes and airmasses of the target during the night
        self.night_altitudes = self.night_altaz_frame.alt.deg
        self.night_airmasses = self.night_altaz_frame.secz
        # Update the altitudes and airmasses for the observation timerange
        self.update_alt_airmass()
        # Get the minimum altitude of the target during the night
        # TODO Think about where the global default setting would live. Thinks like the observer,
        # Telescope pointing limits, etc. These things should be set modifiable by the user, but
        # also have a default value that can be used if the user doesn't want to set them.
        tel_alt_lower_lim = config.scheduling_defaults["telescope elevation limits"][0]  # type: ignore
        tel_alt_upper_lim = config.scheduling_defaults["telescope elevation limits"][1]  # type: ignore
        self.min_altitude = max(self.night_altitudes.min(), tel_alt_lower_lim)
        # Get the maximum altitude of the target during the night
        self.max_altitude = min(self.night_altitudes.max(), tel_alt_upper_lim)

        # Get time of maximum altitude
        # Create a time range for the culmination window (defined in the Night instance)
        culmination_window_timerange = np.linspace(
            Time(self.night.culmination_window[0], format="jd"),
            Time(self.night.culmination_window[1], format="jd"),
            300,
        )
        # Convert to AltAz frame and get the time of maximum altitude
        self.culmination_time = self.target.coords.transform_to(
            self.night.observer.altaz(time=culmination_window_timerange)
        )
        self.culmination_time = culmination_window_timerange[
            np.argmax(self.culmination_time.alt.deg)
        ].jd

        # Get the rise and set times of the target
        start_time_astropy = Time(self.night.obs_within_limits[0], format="jd")
        start_time_astropy = self.night.night_time_range[0]
        if self.night_altitudes[0] > tel_alt_lower_lim:
            # If the target is already up by night start, the rise time is "previous"
            self.rise_time = self.night.observer.target_rise_time(
                start_time_astropy,
                self.target.coords,
                horizon=tel_alt_lower_lim * u.deg,
                which="previous",
                n_grid_points=10,
            ).jd
            # and set time is "next"
            self.set_time = self.night.observer.target_set_time(
                start_time_astropy,
                self.target.coords,
                horizon=tel_alt_lower_lim * u.deg,
                which="next",
                n_grid_points=10,
            ).jd
        else:
            # If the target is not up by night start, the rise time is "next"
            self.rise_time = self.night.observer.target_rise_time(
                start_time_astropy,
                self.target.coords,
                horizon=tel_alt_lower_lim * u.deg,
                which="next",
                n_grid_points=10,
            ).jd
            # and set time is also "next"
            self.set_time = self.night.observer.target_set_time(
                start_time_astropy,
                self.target.coords,
                horizon=tel_alt_lower_lim * u.deg,
                which="next",
                n_grid_points=10,
            ).jd

    def fairness(self) -> float:
        # TODO Implement the fairness
        # fairness_value = np.prod([merit.evaluate() for merit in self.target.fairness_merits])
        fairness_value = 1.0
        return fairness_value  # type: ignore

    def update_alt_airmass(self):
        # Find indices
        night_range_jd = self.night.night_time_range.value
        start_idx = np.searchsorted(night_range_jd, self.start_time, side="left")
        end_idx = np.searchsorted(night_range_jd, self.end_time, side="right")
        self.obs_time_range = night_range_jd[start_idx:end_idx]
        self.obs_altitudes = self.night_altitudes[start_idx:end_idx]
        self.obs_airmasses = self.night_airmasses[start_idx:end_idx]

    def feasible(self, verbose: bool = False) -> float:
        # Update all veto merits and return their product
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

    def separation_calc(self, previous_obs):
        prev_coords = (previous_obs.target.ra_deg, previous_obs.target.dec_deg)
        obs_corrds = (self.target.ra_deg, self.target.dec_deg)
        return np.sqrt(
            (prev_coords[0] - obs_corrds[0]) ** 2 + (prev_coords[1] - obs_corrds[1]) ** 2
        )

    def update_start_time(self, previous_obs):
        # Update of the start time according to slew times and instrument configuration
        # sep_deg = previous_obs.target.coords.separation(self.target.coords).deg

        sep_deg = self.separation_calc(previous_obs)
        # FIX: This is a dummy slew time and inst change for now, replace with fetching from config file
        slew_rate = 2  # degrees per second
        slew_time = (sep_deg / slew_rate) / 86400  # in days
        cor_readout = 0.0002315  # in days
        inst_change = 0.0
        if previous_obs.target.program.instrument != self.target.program.instrument:
            # If the instrument is different, then there is a configuration time
            inst_change = 0.00196759  # 170 seconds in days
        self.start_time = previous_obs.end_time + slew_time + cor_readout + inst_change
        self.end_time = self.start_time + self.exposure_time
        if self.end_time > self.night.obs_within_limits[1]:
            # Set score to zero if observation goes beyond the end of the night
            self.score = 0.0
        else:
            # Update the time array becaue the start time changed
            self.update_alt_airmass()
            # Calculate new rank score based on new start time
            self.evaluate_score()

    def evaluate_score(self, verbose: bool = False) -> float:
        # Run the full rank function for the observation (dummy score for now)
        # This involves running all the veto and efficiency merits.

        # --- Fairness ---
        # Balances time allocation and priority
        fairness = self.fairness_value

        # --- Sensibility ---
        # Veto merits that check for observatbility
        sensibility = self.sensibility_value
        if sensibility == 0.0:
            # If sensibility is zero, then the observation is not feasible
            self.score = 0.0
            return self.score

        # --- Efficiency ---
        # Efficiency merits that check for scientific goal
        efficiency = np.mean([merit.evaluate(self) for merit in self.target.efficiency_merits])

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
        return self.unique_id == other.unique_id  # Or some other unique attribute


class Plan:
    def __init__(self):
        self.observations = []
        self.score = 0.0
        self.evaluation = 0.0

    def add_observation(self, observation: Observation):
        self.observations.append(observation)
        return self

    def calculate_overhead(self):
        # Calculate the overheads for the plan
        # Go through all observation and count the time between the end of one observation and the
        # start of the next one
        first_obs = self.observations[0]
        observation_time = 0.0  # first_obs.exposure_time
        for i, obs in enumerate(self.observations):
            observation_time += obs.exposure_time
        # Check that overhead and observation time add up to the total time
        overhead_time = self.observations[-1].end_time - first_obs.start_time - observation_time
        available_obs_time = (
            first_obs.night.obs_within_limits[1] - first_obs.night.obs_within_limits[0]
        )
        unused_time = available_obs_time - observation_time - overhead_time
        self.observation_time = TimeDelta(observation_time * u.day).to_datetime()
        self.overhead_time = TimeDelta(overhead_time * u.day).to_datetime()
        self.unused_time = TimeDelta(unused_time * u.day).to_datetime()
        self.overhead_ratio = overhead_time / observation_time
        self.observation_ratio = observation_time / available_obs_time

    def evaluate_plan(self) -> float:
        # Evaluate the whole observation plan
        self.score = float(
            np.mean([obs.score for obs in self.observations]) if len(self) > 0 else 0
        )  # type: ignore
        self.calculate_overhead()
        self.evaluation = self.score * self.observation_ratio
        return self.evaluation

    def print_stats(self):
        self.evaluate_plan()
        self.calculate_overhead()
        print(f"Length = {len(self)}")
        print(f"Score = {self.score:.6f}")
        print(f"Evaluation = {self.evaluation:.6f}")
        print(f"Overhead time = {self.overhead_time}")
        print(f"Overhead ratio = {self.overhead_ratio:.5f}")
        print(f"Observation time = {self.observation_time}")
        print(f"Observation ratio = {self.observation_ratio:.5f}")

    def plot(self, display: bool = True, save: bool = False, path: str = None):
        """Plot the schedule for the night."""
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
        ax1.fill_betweenx(y_range, sunset.datetime, civil_evening, color="yellow", alpha=0.2)
        ax1.fill_betweenx(y_range, civil_evening, nautical_evening, color="orange", alpha=0.2)
        ax1.fill_betweenx(y_range, nautical_evening, astronomical_evening, color="red", alpha=0.2)
        # Same for the morning
        ax1.fill_betweenx(y_range, civil_morning, sunrise.datetime, color="yellow", alpha=0.2)
        ax1.fill_betweenx(y_range, nautical_morning, civil_morning, color="orange", alpha=0.2)
        ax1.fill_betweenx(y_range, astronomical_morning, nautical_morning, color="red", alpha=0.2)
        # Add text that have the words "civil", "nautical", and "astronomical".
        # These boxes are placed vertically at the times of each of them (both evening and morning)
        text_kwargs = {
            "rotation": 90,
            "verticalalignment": "bottom",
            "color": "gray",
            "fontsize": 8,
        }

        ax1.text(sunset.datetime, 30.5, "Sunset", horizontalalignment="right", **text_kwargs)
        ax1.text(civil_evening, 30.5, "Civil", horizontalalignment="right", **text_kwargs)
        ax1.text(nautical_evening, 30.5, "Nautical", horizontalalignment="right", **text_kwargs)
        ax1.text(
            astronomical_evening, 30.5, "Astronomical", horizontalalignment="right", **text_kwargs
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
        corresponding_altitudes = list(90.0 - np.degrees(np.arccos(1.0 / desired_airmasses[:-1])))
        corresponding_altitudes.append(90.0)

        # Create the secondary y-axis for Airmass
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())
        # Set y-ticks at computed altitudes for desired airmasses
        ax2.set_yticks(corresponding_altitudes)
        ax2.set_yticklabels(np.round(desired_airmasses, 2))  # Display the desired airmass values
        ax2.set_ylabel("Airmass")
        ax2.tick_params("y")
        if save:
            plt.tight_layout()
            if path is None:
                raise ValueError("Path must be specified if save is True")
            else:
                plt.savefig(path, dpi=300)
        if display:
            plt.show()
        else:
            plt.close()

    def plot_interactive(self, save: bool = False, path: str = None):
        """Plot the schedule for the night using Plotly."""
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
                np.linspace(obs.start_time, (obs.start_time + obs.exposure_time), 20), format="jd"
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

            # # Hover text
            # hover_text = (
            #     f"{obs.target.name}\n\n{obs.target.program.instrument} {program_id}\n"
            #     + f"Obs start:{Time(obs.start_time, format='jd').datetime}\n"
            #     + f"Exp time:{TimeDelta(obs.exposure_time * u.day).to_datetime()}"
            # )
            hovertemplate = (
                f"<b>{obs.target.name}</b><br>"
                + f"{obs.target.program.instrument} {program_id}<br><br>"
                + f"Start time: {Time(obs.start_time, format='jd').datetime.strftime('%H:%M:%S %d-%m-%Y')}<br>"
                + f"Exp time: {TimeDelta(obs.exposure_time * u.day).to_datetime()}"
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
        corresponding_altitudes = list(90.0 - np.degrees(np.arccos(1.0 / desired_airmasses[:-1])))
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
        lines = []
        lines.append(
            f"Plan for the night of {self.observations[0].night.night_date} (Times in UTC)"
        )
        lines.append("--------------------------------------------------\n")
        # lines.append(" #      Program ID      Target                  Start time   (Exp time)")
        lines.append(
            f"{'#':<6}{'Program ID':<12}{'Target':<20}{'Start time':<13}{'(Exp time)':<10}"
        )
        for i, obs in enumerate(self.observations):
            start_time = Time(obs.start_time, format="jd").datetime
            exp_time = TimeDelta(obs.exposure_time * u.day).to_datetime()
            string = f"{f'{i+1:>2}:':<6}"
            string += f"{f'{obs.target.program.progID} {obs.target.program.instrument}':<12}"
            string += f"{obs.target.name:<20}"
            string += f"{start_time.strftime('%H:%M:%S'):<13}"
            string += f"{f'({exp_time})':<10}"
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
