from typing import Dict, Any, List, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta, TimeJD
from datetime import date, datetime, timedelta
import astropy.units as u
from astroplan import Observer
import inspect
import config

# Define global observer location
lasilla = Observer.at_site("lasilla")

class Night:
    def __init__(self, night_date: date, observer: Observer = lasilla):
        # How to calculate which night to take from the input time?
        # For now just take the night of the input time
        self.night_date = night_date
        self.observer = observer
        self.midnight = Time(datetime.combine(self.night_date + timedelta(days=1), datetime.min.time()))
        self.sunset = self.observer.sun_set_time(self.midnight, which='previous')
        self.sunrise = lasilla.sun_rise_time(self.midnight, which='next')

        # Get the times for the different twilights
        self.civil_evening = lasilla.twilight_evening_civil(self.midnight, which='previous')
        self.nautical_evening = lasilla.twilight_evening_nautical(self.midnight, which='previous')
        self.astronomical_evening = lasilla.twilight_evening_astronomical(self.midnight, which='previous')
        # And the same for the morning
        self.civil_morning = lasilla.twilight_morning_civil(self.midnight, which='next')
        self.nautical_morning = lasilla.twilight_morning_nautical(self.midnight, which='next')
        self.astronomical_morning = lasilla.twilight_morning_astronomical(self.midnight, which='next')
        # Time ranges for the different twilights
        self.time_range_night = np.linspace(self.sunset, self.sunrise, 300)
        self.time_range_civil = np.linspace(self.civil_evening, self.civil_morning, 300)
        self.time_range_nautical = np.linspace(self.nautical_evening, self.nautical_morning, 300)
        self.time_range_astronomical = np.linspace(self.astronomical_evening, self.astronomical_morning, 300)

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
        self.time_share_pct_diff = (self.time_share_current-self.time_share_allocated)/self.time_share_allocated


class Merit:
    def __init__(self, name: str, func: Callable, merit_type: str, parameters: Dict[str, Any] = {}):
        self.name = name
        self.func = func  # The function that computes this merit
        self.description = self.func.__doc__
        self.merit_type = merit_type  # "veto" or "efficiency"
        self.parameters = parameters  # Custom parameters for this merit

        # Check that the merit type is valid
        if self.merit_type not in ["fairness", "veto", "efficiency"]:
            raise ValueError(f"Invalid merit type ({self.merit_type}). "
                             "Valid types are 'fairness', 'veto' and 'efficiency'.")

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
        assert required_func_parameters[0] == "observation", "The first parameter has to be 'observation'" 
        assert set(required_func_parameters[1:]).issubset(set(self.parameters.keys())), (
              f"The given parameters ({set(self.parameters.keys())}) don't match the "
               "required parameters of the given function "
               f"({set(required_func_parameters[1:])})"
               )
        assert set(self.parameters.keys()).issubset(set(required_func_parameters+optional_func_parameters)), (
            f"There are given parameters that are not part of the given function"
        )

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
        self.last_obs = last_obs
        self.merits: List[Merit] = []  # List of all merits
    
    def add_merit(self, merit: Merit):
        self.merits.append(merit)

    def __str__(self):
        lines = ["Target(Name: {self.name},",
                 f"       Program: {self.program.progID},",
                 f"       Coordinates: {self.coords},",
                 f"       Last observation: {self.last_obs},",
                 f"       Merits: {self.merits},",
                 f"       Time share allocated: {self.program.time_share_allocated},",
                 f"       Time share current: {self.program.time_share_current},",
                 f"       Time share pct diff: {self.program.time_share_pct_diff})"
                 ]
        
        return "\n".join(lines)
        


class Observation:
    def __init__(self, target: Target, start_time: Time, exposure_time: TimeDelta, 
                 night: Night, observer: Observer = lasilla):
        self.target = target
        self.start_time = start_time
        self.exposure_time = exposure_time
        self.night = night
        self.observer = observer  # Observer location
        self.score: float = 0.0  # Initialize score to zero
        self.veto_merits: List[float] = []  # List to store veto merits
        self.time_array: Time = None

        # Initizalize the time array with the start time
        self.update_time_array()


        # Calculate the minimum and maximum altitude of the target during the night of the observation
        # Create the AltAz frame for the observation during the night
        # Check for an at_night merit
        if any(merit.func.__name__ == "at_night" for merit in self.target.merits):
            # Get the at_night merit
            at_night_merit = [merit for merit in self.target.merits if merit.merit_type == "veto" and merit.func.__name__ == "at_night"][0]
            # Get the which parameter of the at_night merit
            twilight = at_night_merit.parameters['which']
        else:
            twilight = 'astronomical'
        
        # Get the time range for the night
        if twilight == 'civil':
            time_range = self.night.time_range_civil
        elif twilight == 'nautical':
            time_range = self.night.time_range_nautical
        elif twilight == 'astronomical':
            time_range = self.night.time_range_astronomical

        # Create the AltAz frame for the observation during the night
        self.night_altaz_frame = self.observer.altaz(time=time_range)
        # Get the altitude of the target during the night
        self.night_altitudes = self.target.coords.transform_to(self.night_altaz_frame).alt.deg
        # Get the minimum altitude of the target during the night
        self.min_altitude = max(self.night_altitudes.min(), 
                                config.scheduling_defaults['telescope elevation limits'][0]) # type: ignore
        # Get the maximum altitude of the target during the night
        self.max_altitude = min(self.night_altitudes.max(), 
                                config.scheduling_defaults['telescope elevation limits'][1]) # type: ignore

    
    def update_time_array(self):
        # Create time range for the observation
        self.time_array = np.linspace(self.start_time, (self.start_time + self.exposure_time), 10)
        self.observation_altaz_frame = self.observer.altaz(time=self.time_array)
        self.coords_altaz = self.target.coords.transform_to(self.observation_altaz_frame)

    def update_veto_merits(self):
        # Update all veto merits (dummy values for now)
        self.veto_merits = [merit.evaluate(self) for merit in self.target.merits if merit.merit_type == "veto"]

    def update_start_time(self, observation):
        # Update of the start time according to slew times and instrument configuration
        # For now it just adds the exposure time, assuming no overheads
        # TODO Implement the overheads of slew time from previous observation and instrument configuration
        self.start_time = observation.start_time + observation.exposure_time
        
        # Update the time array becaue the start time changed
        self.update_time_array()

    def feasible(self) -> float:
        # Update all veto merits (dummy values for now) and return their product
        self.update_veto_merits()

        return np.prod(self.veto_merits) # type: ignore

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
        efficiency = np.mean([merit.evaluate(self) for merit in self.target.merits if merit.merit_type == "efficiency"])


        # --- Rank Score ---
        # Calculate total rank score by taking the product of fairness, sensibility and efficiency
        self.score = np.prod([fairness, sensibility, efficiency]) # type: ignore
        
        # Print results if verbose
        if verbose:
            print(f"Fairness: {fairness}")
            print(f"Sensibility: {sensibility}")
            print(f"Efficiency: {efficiency}")
            print(f"Rank score: {self.score}")

        return self.score
    


    def __str__(self):
        lines = [f"Observation(Target: {self.target.name},",
                 f"            Start time: {self.start_time},",
                 f"            Exposure time: {self.exposure_time},",
                 f"            Score: {self.score})"
                 ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.__str__()
    


class Plan:
    def __init__(self, observation_night: Time):
        self.observations: List[Observation] = []
        self.score: float = 0.0
        # TODO: rethink what the best type class for the observation_night should be
        # If a simple Time object with the time and date of the sunset
        # Or just a date, or julian date, or something else
        self.observation_night = observation_night
        
    def add_observation(self, observation: Observation):
        self.observations.append(observation)
        
    def evaluate_plan(self) -> float:    
        # Evaluate the whole observation plan
        self.score = np.mean([obs.score for obs in self.observations]) if len(self) > 0 else 0 # type: ignore
        # TODO Add some other metrics, like the total time used, minimization of overheads, etc.
        return self.score
    
    def plot(self, save: bool = False):
        """ Plot the schedule for the night. """
        first_obs = self.observations[0]

        # Get sunset and sunrise times for this night
        sunset = first_obs.observer.sun_set_time(first_obs.start_time, which='previous')
        sunrise = first_obs.observer.sun_rise_time(first_obs.start_time, which='next')

        # Get the times for the different twilights
        civil_evening = lasilla.twilight_evening_civil(first_obs.start_time, which='previous')
        nautical_evening = lasilla.twilight_evening_nautical(first_obs.start_time, which='previous')
        astronomical_evening = lasilla.twilight_evening_astronomical(first_obs.start_time, which='previous')
        # And the same for the morning
        civil_morning = lasilla.twilight_morning_civil(first_obs.start_time, which='next')
        nautical_morning = lasilla.twilight_morning_nautical(first_obs.start_time, which='next')
        astronomical_morning = lasilla.twilight_morning_astronomical(first_obs.start_time, which='next')

        plt.figure(figsize=(13, 5))

        for obs in self.observations:
            time_range = np.linspace(obs.start_time, (obs.start_time + obs.exposure_time), 20).datetime
            altitudes = obs.target.coords.transform_to(lasilla.altaz(time=time_range)).alt.deg

            # Generate an array of equally spaced Julian Dates
            num_points = 300  # The number of points you want
            jd_array = np.linspace(sunset.jd, sunrise.jd, num_points)

            # Convert back to Astropy Time objects
            night_time_array = Time(jd_array, format='jd', scale='utc').datetime

            night_altitudes = obs.target.coords.transform_to(lasilla.altaz(time=night_time_array)).alt.deg


            # plot the target
            plt.plot_date(night_time_array, night_altitudes, '-.', color='gray', alpha=0.6, linewidth=0.3)
            plt.plot_date(time_range, altitudes, 'k-', linewidth=2, clip_box=True)

        # Plot shaded areas between sunset and civil, nautical, and astronomical evening
        y_range = np.arange(0, 91)
        plt.fill_betweenx(y_range, sunset.datetime, civil_evening.datetime, color='yellow', alpha=0.2)
        plt.fill_betweenx(y_range, civil_evening.datetime, nautical_evening.datetime, color='orange', alpha=0.2)
        plt.fill_betweenx(y_range, nautical_evening.datetime, astronomical_evening.datetime, color='red', alpha=0.2)
        # Same for the morning
        plt.fill_betweenx(y_range, civil_morning.datetime, sunrise.datetime, color='yellow', alpha=0.2)
        plt.fill_betweenx(y_range, nautical_morning.datetime, civil_morning.datetime, color='orange', alpha=0.2)
        plt.fill_betweenx(y_range, astronomical_morning.datetime, nautical_morning.datetime, color='red', alpha=0.2)
        # Add text boxes that have the words "civil", "nautical", and "astronomical". These boxes are placed
        # vertically at the times of each of them (both evening and morning)
        text_kwargs = {'rotation':90, 'verticalalignment':'bottom', 'color':'gray', 'fontsize':8}
        # plt.text(civil_evening.datetime, 51, "Civil", rotation=90, verticalalignment='bottom', horizontalalignment='right', color='gray', fontsize=8)
        plt.text(sunset.datetime, 30.5, "Sunset", horizontalalignment='right', **text_kwargs)
        plt.text(civil_evening.datetime, 30.5, "Civil", horizontalalignment='right', **text_kwargs)
        plt.text(nautical_evening.datetime, 30.5, "Nautical", horizontalalignment='right', **text_kwargs)
        plt.text(astronomical_evening.datetime, 30.5, "Astronomical", horizontalalignment='right', **text_kwargs)
        plt.text((civil_morning+3*u.min).datetime, 30.5, "Civil", horizontalalignment='left', **text_kwargs)
        plt.text((nautical_morning+3*u.min).datetime, 30.5, "Nautical", horizontalalignment='left', **text_kwargs)
        plt.text((astronomical_morning+3*u.min).datetime, 30.5, "Astronomical", horizontalalignment='left', **text_kwargs)
        plt.text((sunrise+3*u.min).datetime, 30.5, "Sunrise", horizontalalignment='left', **text_kwargs)

        # Use DateFormatter to format x-axis to only show time
        time_format = mdates.DateFormatter('%H:%M:%S')
        plt.gca().xaxis.set_major_formatter(time_format)

        # In the title put the date of the schedule
        plt.title(f"Schedule for the night of {self.observation_night.datetime.date()}")

        plt.xlabel("Time [UTC]")
        plt.ylabel("Altitude [deg]")
        plt.ylim(30, 90)
        if save:
            plt.tight_layout()
            plt.savefig(f"night_schedule_{self.observation_night.datetime.date()}.png", dpi=300)
    
    def __len__(self):
        return len(self.observations)
    
    def __str__(self):
        lines = ["Plan(",
                 f"     Score: {self.score},",
                 f"     Observations: ",
                 f"         {self.observations})"
                 ]
        
        return "\n".join(lines)
    
    