import inspect
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List

import astropy.units as u
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astroplan import Observer
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

import config

# Define global observer location
lasilla = Observer.at_site("lasilla")


class Night:
    def __init__(self, night_date: date, observer: Observer = lasilla):
        # How to calculate which night to take from the input time?
        # For now just take the night of the input time
        self.night_date = night_date
        self.observer = observer
        self.midnight = Time(
            datetime.combine(self.night_date, datetime.min.time()) + timedelta(days=1, hours=4)
        )
        self.sunset = self.observer.sun_set_time(self.midnight, which="previous")
        self.sunrise = lasilla.sun_rise_time(self.midnight, which="next")

        # Get the times for the different twilights
        self.civil_evening = lasilla.twilight_evening_civil(self.midnight, which="previous")
        self.nautical_evening = lasilla.twilight_evening_nautical(self.midnight, which="previous")
        self.astronomical_evening = lasilla.twilight_evening_astronomical(
            self.midnight, which="previous"
        )
        # And the same for the morning
        self.civil_morning = lasilla.twilight_morning_civil(self.midnight, which="next")
        self.nautical_morning = lasilla.twilight_morning_nautical(self.midnight, which="next")
        self.astronomical_morning = lasilla.twilight_morning_astronomical(
            self.midnight, which="next"
        )
        # Time ranges for the different twilights
        self.time_range_night = np.linspace(self.sunset, self.sunrise, 300)
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


class Program:
    def __init__(self, progID: int, time_share_allocated: float, instrument: str):
        self.progID = progID
        self.instrument = instrument
        self.time_share_allocated = time_share_allocated
        self.time_share_current = 0.0
        self.time_share_pct_diff = 0.0

    def update_time_share(self):
        # Function that retrieves the current time used by the program
        # TODO: Implement the retrieval of current time shares
        self.time_share_pct_diff = (
            self.time_share_current - self.time_share_allocated
        ) / self.time_share_allocated


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
        return f"Merit({self.name}, {self.merit_type})"

    def __repr__(self):
        return f"Merit({self.name}, {self.merit_type})"


class Target:
    def __init__(self, name: str, prog: Program, coords: SkyCoord, last_obs: Time):
        self.name = name
        self.program = prog
        self.coords = coords
        self.last_obs = last_obs.jd
        self.merits: List[Merit] = []  # List of all merits

    def add_merit(self, merit: Merit):
        self.merits.append(merit)

    def __str__(self):
        lines = [
            "Target(Name: {self.name},",
            f"       Program: {self.program.progID},",
            f"       Coordinates: {self.coords},",
            f"       Last observation: {self.last_obs},",
            f"       Merits: {self.merits},",
            f"       Time share allocated: {self.program.time_share_allocated},",
            f"       Time share current: {self.program.time_share_current},",
            f"       Time share pct diff: {self.program.time_share_pct_diff})",
        ]

        return "\n".join(lines)


class Observation:
    def __init__(
        self,
        target: Target,
        start_time: Time,
        exposure_time: TimeDelta,
        night: Night,
        observer: Observer = lasilla,
    ):
        self.target = target
        self.start_time = start_time.jd
        self.exposure_time = exposure_time.value
        self.night = night
        self.observer = observer  # Observer location
        self.score: float = 0.0  # Initialize score to zero
        self.veto_merits: List[float] = []  # List to store veto merits
        # self.time_array: Time = None
        self.unique_id = uuid.uuid4()

        # Calculate the minimum and maximum altitude of the target during the night of the observation
        # Create the AltAz frame for the observation during the night
        # Check for an at_night merit
        if any(merit.func.__name__ == "at_night" for merit in self.target.merits):
            # Get the at_night merit
            at_night_merit = [
                merit
                for merit in self.target.merits
                if merit.merit_type == "veto" and merit.func.__name__ == "at_night"
            ][0]
            # Get the which parameter of the at_night merit
            twilight = at_night_merit.parameters["which"]
        else:
            twilight = "astronomical"

        # Get the time range for the night
        if twilight == "civil":
            self.night_time_range = self.night.time_range_civil
        elif twilight == "nautical":
            self.night_time_range = self.night.time_range_nautical
        elif twilight == "astronomical":
            self.night_time_range = self.night.time_range_astronomical

        # Create the AltAz frame for the observation during the night
        self.night_altaz_frame = self.target.coords.transform_to(
            self.observer.altaz(time=self.night_time_range)
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

        # Figure out the rise and set times for this target
        # But.. what if the target is not up at the current start time?
        # Then the rise time will be the next (will be up soon) or previous (already set and will not)
        # rise until the end of the night
        # First find which is the nearest to current time (rise of set)

        nearest_rise_time = observer.target_rise_time(
            start_time,
            self.target.coords,
            horizon=tel_alt_lower_lim * u.deg,
            which="nearest",
        ).jd
        nearest_set_time = observer.target_set_time(
            start_time,
            self.target.coords,
            horizon=tel_alt_lower_lim * u.deg,
            which="nearest",
        ).jd
        # Calculate timedelta between the start time and the nearest rise and set times
        rise_timedelta = abs(nearest_rise_time - self.start_time)
        set_timedelta = abs(nearest_set_time - self.start_time)
        if rise_timedelta < set_timedelta:
            # If the rise time is closer than the set time, then the target is rising
            self.rise_time = nearest_rise_time
            self.set_time = observer.target_set_time(
                start_time,
                self.target.coords,
                horizon=tel_alt_lower_lim * u.deg,
                which="next",
            ).jd
        elif set_timedelta < rise_timedelta:
            # If the set time is closer than the rise time, then the target already set
            self.rise_time = observer.target_rise_time(
                start_time,
                self.target.coords,
                horizon=tel_alt_lower_lim * u.deg,
                which="previous",
            ).jd
            self.set_time = nearest_rise_time

    def update_alt_airmass(self):
        # Find indices
        night_range_jd = self.night_time_range.value
        start_idx = np.searchsorted(night_range_jd, self.start_time, side="left")
        end_idx = np.searchsorted(
            night_range_jd, self.start_time + self.exposure_time, side="right"
        )
        self.obs_altitudes = self.night_altitudes[start_idx:end_idx]
        self.obs_airmasses = self.night_airmasses[start_idx:end_idx]

    def update_veto_merits(self):
        # Update all veto merits (dummy values for now)
        self.veto_merits = [
            merit.evaluate(self) for merit in self.target.merits if merit.merit_type == "veto"
        ]

    def feasible(self) -> float:
        # Update all veto merits (dummy values for now) and return their product
        self.update_veto_merits()

        return np.prod(self.veto_merits)  # type: ignore

    def update_start_time(self, previous_obs):
        # Update of the start time according to slew times and instrument configuration
        # For now it just adds the exposure time, assuming no overheads
        # TODO Implement the overheads of slew time from previous observation and instrument configuration
        self.start_time = previous_obs.start_time + previous_obs.exposure_time
        # Update the time array becaue the start time changed
        self.update_alt_airmass()
        # Calculate new rank score based on new start time
        self.evaluate_score()

    def evaluate_score(self, verbose: bool = False) -> float:
        # Run the full rank function for the observation (dummy score for now)
        # This involves running all the veto and efficiency merits.

        # --- Fairness ---
        # Balances time allocation and priority
        # TODO Implement the fairness
        fairness: float = 1

        # --- Sensibility ---
        # Veto merits that check for observatbility
        sensibility = self.feasible()

        # --- Efficiency ---
        # Efficiency merits that check for scientific goal
        efficiency = np.mean(
            [
                merit.evaluate(self)
                for merit in self.target.merits
                if merit.merit_type == "efficiency"
            ]
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
        return self.unique_id == other.unique_id  # Or some other unique attribute


class Plan:
    def __init__(self):
        self.observations = []
        self.score = 0.0
        # TODO: rethink what the best type class for the observation_night should be
        # If a simple Time object with the time and date of the sunset
        # Or just a date, or julian date, or something else

    def add_observation(self, observation: Observation):
        self.observations.append(observation)
        return self

    def evaluate_plan(self) -> float:
        # Evaluate the whole observation plan
        self.score = float(
            np.mean([obs.score for obs in self.observations]) if len(self) > 0 else 0
        )  # type: ignore
        # TODO Add some other metrics, like the total time used, minimization of overheads, etc.
        return self.score

    def plot(self, save: bool = False):
        """Plot the schedule for the night."""
        first_obs = self.observations[0]

        # Get sunset and sunrise times for this night
        night = first_obs.night
        sunset = Time(night.sunset, format='jd')
        sunrise = Time(night.sunrise, format='jd')

        # Get the times for the different twilights
        civil_evening = Time(night.civil_evening, format='jd').datetime
        nautical_evening = Time(night.nautical_evening, format='jd').datetime
        astronomical_evening = Time(night.astronomical_evening, format='jd').datetime
        civil_morning = Time(night.civil_morning, format='jd').datetime
        nautical_morning = Time(night.nautical_morning, format='jd').datetime
        astronomical_morning = Time(night.astronomical_morning, format='jd').datetime

        # Get which programs are part of this plan
        programs = list(
            set([obs.target.program.progID for obs in self.observations])
        )  # Remove duplicates
        # Define unique colors for each program
        # Use the 'Set2' color pallete from matplotlib
        colors = plt.get_cmap("Set2").colors
        # Create a dictionary that maps each program to a color
        prog_colors = dict(zip(programs, colors))

        plt.figure(figsize=(13, 5))

        for obs in self.observations:
            time_range = Time(np.linspace(
                obs.start_time, (obs.start_time + obs.exposure_time), 20
            ), format='jd').datetime
            altitudes = obs.target.coords.transform_to(lasilla.altaz(time=time_range)).alt.deg

            # Generate an array of equally spaced Julian Dates
            num_points = 300  # The number of points you want
            jd_array = np.linspace(sunset.jd, sunrise.jd, num_points)

            # Convert back to Astropy Time objects
            night_time_array = Time(jd_array, format="jd", scale="utc").datetime

            night_altitudes = obs.target.coords.transform_to(
                lasilla.altaz(time=night_time_array)
            ).alt.deg

            # plot the target
            plt.plot_date(
                night_time_array,
                night_altitudes,
                "-.",
                color="gray",
                alpha=0.6,
                linewidth=0.3,
            )
            plt.plot_date(
                time_range,
                altitudes,
                "-",
                c=prog_colors[obs.target.program.progID],
                lw=2,
                solid_capstyle="round",
            )

        # Plot shaded areas between sunset and civil, nautical, and astronomical evening
        y_range = np.arange(0, 91)
        plt.fill_betweenx(y_range, sunset.datetime, civil_evening, color="yellow", alpha=0.2)
        plt.fill_betweenx(y_range, civil_evening, nautical_evening, color="orange", alpha=0.2)
        plt.fill_betweenx(y_range, nautical_evening, astronomical_evening, color="red", alpha=0.2)
        # Same for the morning
        plt.fill_betweenx(y_range, civil_morning, sunrise.datetime, color="yellow", alpha=0.2)
        plt.fill_betweenx(y_range, nautical_morning, civil_morning, color="orange", alpha=0.2)
        plt.fill_betweenx(y_range, astronomical_morning, nautical_morning, color="red", alpha=0.2)
        # Add text that have the words "civil", "nautical", and "astronomical".
        # These boxes are placed vertically at the times of each of them (both evening and morning)
        text_kwargs = {
            "rotation": 90,
            "verticalalignment": "bottom",
            "color": "gray",
            "fontsize": 8,
        }

        plt.text(sunset.datetime, 30.5, "Sunset", horizontalalignment="right", **text_kwargs)
        plt.text(civil_evening, 30.5, "Civil", horizontalalignment="right", **text_kwargs)
        plt.text(nautical_evening, 30.5, "Nautical", horizontalalignment="right", **text_kwargs)
        plt.text(
            astronomical_evening, 30.5, "Astronomical", horizontalalignment="right", **text_kwargs
        )
        plt.text(
            (civil_morning + timedelta(minutes=3)),
            30.5,
            "Civil",
            horizontalalignment="left",
            **text_kwargs,
        )
        plt.text(
            (nautical_morning + timedelta(minutes=3)),
            30.5,
            "Nautical",
            horizontalalignment="left",
            **text_kwargs,
        )
        plt.text(
            (astronomical_morning + timedelta(minutes=3)),
            30.5,
            "Astronomical",
            horizontalalignment="left",
            **text_kwargs,
        )
        plt.text(
            (sunrise.datetime + timedelta(minutes=3)),
            30.5,
            "Sunrise",
            horizontalalignment="left",
            **text_kwargs,
        )

        # Use DateFormatter to format x-axis to only show time
        time_format = mdates.DateFormatter("%H:%M:%S")
        plt.gca().xaxis.set_major_formatter(time_format)

        # In the title put the date of the schedule
        plt.title(f"Schedule for the night of {self.observations[0].night.night_date}")

        plt.xlabel("Time [UTC]")
        plt.ylabel("Altitude [deg]")
        plt.ylim(30, 90)
        if save:
            plt.tight_layout()
            plt.savefig(f"night_schedule_{self.observations[0].night.night_date}.png", dpi=300)

    def __len__(self):
        return len(self.observations)

    def __str__(self):
        lines = [
            "Plan(",
            f"     Score: {self.score},",
            "     Observations: ",
            f"         {self.observations})",
        ]

        return "\n".join(lines)
