from class_definitions import Program, Merit, Target, Observation, Plan
from typing import Dict, Any, List, Callable
from astropy.time import Time
import numpy as np

def get_observation_night(observation: Observation) -> Time:
    """ Return the night of an observation. """
    sunset = observation.observer.sun_set_time(observation.start_time, which='previous')
    return sunset.date()

def update_start_times(observations: List[Observation], previous_obs: Observation):
    """ Update the start time of all observations in the list based on the previous observation. """
    for obs in observations:
        obs.update_start_time(previous_obs)

def forwardP(start_obs: Observation, available_observations: List[Observation], lookahead_distance: int=None):
    """ Basic scheduler that simply continues a Plan from the starting observation by
     sequentially choosing the highest scoring observation. """
    
    # Set the lookahead distance to the number of available observations if not specified
    # Or to finish the night if there are more available observations than time in the night
    if (lookahead_distance is not None) and (len(available_observations) <= lookahead_distance):
        raise ValueError(f"Number of available observations ({len(available_observations)}) "
                         f"must be less than or equal to lookahead distance ({lookahead_distance})")
    elif lookahead_distance is None:
        lookahead_distance = len(available_observations)
    
    # Get observation night from the starting observation
    sunset = start_obs.observer.sun_set_time(start_obs.start_time, which='previous')
    sunrise = start_obs.observer.sun_rise_time(start_obs.start_time, which='next')

    # Initialize the Plan object
    observation_plan = Plan(sunset)
    observation_plan.add_observation(start_obs)
    update_start_times(available_observations, start_obs)

    # Add candidate observation to plan K times
    for _ in range(lookahead_distance):
    
        # Initialize Q as an empty list to store ranked observations
        Q = []

        # Evaluate each available observation
        for o_prime in available_observations:
            if o_prime.feasible():
                score = o_prime.evaluate_score()
                # Insert into Q ensuring Q is sorted by score
                Q.append((score, o_prime))

        # Sort Q by score
        Q.sort(reverse=True, key=lambda x: x[0])

        # Check exit conditions
        if not Q or len(observation_plan) >= lookahead_distance:
            break

        # Select the highest ranking observation
        if Q:
            # Select the highest ranking observation
            _, o_double_prime = Q[0]

            # Add the selected observation to the plan
            observation_plan.add_observation(o_double_prime)

            # Remove the selected observation from the available observations
            available_observations.remove(o_double_prime)

            # Update the start time of all remaining observations
            update_start_times(available_observations, o_double_prime)

            
    
    # Evaluate the plan before returning
    observation_plan.evaluate_plan()

    return observation_plan
