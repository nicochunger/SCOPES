import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, List, Tuple, Union

# from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
from astropy.time import Time

from class_definitions import Observation, Plan


## ----- HELPER FUNCTIONS ----- ##
def pickle_deepcopy(obj):
    return pickle.loads(pickle.dumps(obj, -1))


def obslist_deepcopy(obslist):
    """
    An implementation of deepcoping a list of observations by creating new emtpy observations
    and assigning the attributes of the original observations to the new ones.
    """
    new_obslist = []
    for obs in obslist:
        new_obs = Observation.__new__(Observation)
        new_obs.__dict__ = obs.__dict__.copy()
        new_obslist.append(new_obs)
    return new_obslist


def get_observation_night(observation: Observation) -> Time:
    """Return the night of an observation."""
    sunset = observation.observer.sun_set_time(observation.start_time, which="previous")
    return sunset.date()


def update_start_times(observations: List[Observation], new_start_time: float):
    """Update the start time of all observations in the list based on the previous observation."""
    for obs in observations:
        obs.start_time = new_start_time
        obs.end_time = obs.start_time + obs.exposure_time
        obs.update_alt_airmass()


def update_start_from_prev(observations: List[Observation], previous_obs: Observation):
    """Update the start time of all observations in the list based on the previous observation."""
    for obs in observations:
        obs.update_start_time(previous_obs)


## ----- SCHEDULERS ----- ##


# Basic forward scheduler, greedy search
def forwardP(
    start_obs: Union[Observation, Time],
    available_observations: List[Observation],
    lookahead_distance=None,
):
    """Basic scheduler that simply continues a Plan from the starting observation by
    sequentially choosing the highest scoring observation."""

    # Set the lookahead distance to the number of available observations if not specified
    # Or to finish the night if there are more available observations than time in the night
    if (lookahead_distance is not None) and (len(available_observations) <= lookahead_distance):
        raise ValueError(
            f"Number of available observations ({len(available_observations)}) "
            f"must be less than or equal to lookahead distance ({lookahead_distance})"
        )
    elif lookahead_distance is None:
        # Equivalent to planning until the end of night
        lookahead_distance = len(available_observations)

    # Create deep copy of available observations
    Obs_copy = deepcopy(available_observations)

    # Initialize the Plan object according to if the starting codition is a Time or Observation
    observation_plan = Plan()
    if isinstance(start_obs, Observation):
        observation_plan.add_observation(start_obs)
        update_start_from_prev(Obs_copy, start_obs)
    elif isinstance(start_obs, Time):
        for obs in Obs_copy:
            obs.start_time = start_obs.jd
            obs.end_time = obs.start_time + obs.exposure_time
            obs.update_alt_airmass()
    else:
        raise TypeError(f"start_obs must be of type Observation or Time, not {type(start_obs)}")

    # Add candidate observation to plan K times

    for _ in range(lookahead_distance):
        # Initialize Q as an empty list to store ranked observations
        Q = []

        # Evaluate each available observation
        for o_prime in Obs_copy:
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
            Obs_copy.remove(o_double_prime)

            # Update the start time of all remaining observations
            update_start_from_prev(Obs_copy, o_double_prime)

    # Evaluate the plan before returning
    observation_plan.evaluate_plan()

    return observation_plan


# Dynamic programming scheduler using recursion
class DPPlanner:
    def __init__(self):
        self.DP = {}
        self.total_counter = 0
        self.saved_state_counter = 0

    def dp_recursion(
        self,
        remaining_observations: List[Observation],
        # current_time: Time,
        current_plan: Plan,
        max_plan_length: int,
        K: int = 5,
    ) -> Tuple[float, Plan]:
        """
        Dynamic programming recursive function to find the best plan from a given state.
        It uses beam search to consider only the top K observations at each step.
        """
        self.total_counter += 1
        # Sort remaining_observations by target names
        sorted_remaining_observations = sorted(
            remaining_observations, key=lambda obs: obs.target.name
        )

        # Create the state tuple using sorted remaining observations and current plan length
        state = (
            tuple(obs.unique_id for obs in sorted_remaining_observations),
            len(current_plan),
        )

        # Check if state has already been computed
        if state in self.DP:
            self.saved_state_counter += 1
            return self.DP[state]

        # Base case 1: No remaining observations, evaluate current plan
        # Base case 2: Plan has reached maximum length
        if len(remaining_observations) == 0 or len(current_plan) >= max_plan_length:
            score = current_plan.evaluate_plan()
            self.DP[state] = (score, current_plan)
            return score, current_plan

        # Initialize variables to hold the best score and corresponding plan
        best_score = float("-inf")
        best_plan: Plan = Plan()

        # Loop through remaining observations to consider adding each to the plan
        top_k_observations = sorted(
            remaining_observations, key=lambda x: x.evaluate_score(), reverse=True
        )[:K]
        for obs in top_k_observations:
            # Create a deep copy of current_plan first
            new_plan = deepcopy(current_plan)

            # Create a copy of the remaining observations
            remaining_copy = deepcopy(remaining_observations)
            # Remove the observation
            remaining_copy.remove(obs)

            # Check if adding this observation is feasible
            # NOTE: I think this check can be omitted as its already done in top_k_observations
            if obs.feasible():
                # Add observation to the new plan
                new_plan.add_observation(obs)

                # Update the current time based on the end time of the added observation
                update_start_from_prev(remaining_copy, obs)

                # Recursive call to find best plan from this point forward
                _, temp_plan = self.dp_recursion(remaining_copy, new_plan, max_plan_length, K)

                # Evaluate this complete plan
                score = temp_plan.evaluate_plan()

                # Update best score and plan if this plan is better
                if score > best_score:
                    best_score = score
                    best_plan = temp_plan

        # Store the best score and plan for this state
        self.DP[state] = (best_score, best_plan)

        return best_score, best_plan


# Beam search scheduler
class BeamSearchPlanner:
    def __init__(self, plan_start_time: float = None) -> None:
        self.total_counter = 0
        self.depth = 0
        self.plan_start_time = plan_start_time

    @dataclass(order=True)
    class PrioritizedItem:
        score: float
        plan: Any = field(compare=False)
        obs: Any = field(compare=False)

    def dp_beam_search(
        self, initial_observations: List[Observation], max_plan_length: int = None, K: int = 5
    ) -> Plan:
        # Check if there is a plan start time
        if self.plan_start_time is not None:
            update_start_times(initial_observations, self.plan_start_time)
        if max_plan_length is None:
            max_plan_length = len(initial_observations)

        # Initialize two priority queues
        PQ_current: PriorityQueue = PriorityQueue()
        PQ_next: PriorityQueue = PriorityQueue()

        # Add initial state to the current priority queue
        PQ_current.put(self.PrioritizedItem(0, Plan(), initial_observations))

        # Initialize best plan and best score to None and -inf
        best_plan: Plan = Plan()

        while not PQ_current.empty():
            self.total_counter += 1

            # Retrieve the highest-score plan from the current priority queue
            pq_current_item = PQ_current.get()
            # current_score = pq_current_item.score
            current_plan = pq_current_item.plan
            remaining_observations = pq_current_item.obs

            # Check stopping criteria
            if len(current_plan) >= max_plan_length:
                break

            Q = []
            # Generate child plans by extending the current plan with feasible observations
            for obs in remaining_observations:
                if obs.feasible():
                    score = obs.evaluate_score()
                    Q.append((score, obs))
            if not Q:
                # Termination condition if no feasible observations remain
                break

            # Sort Q by score
            Q.sort(reverse=True, key=lambda x: x[0])
            for _, obs in Q:
                # Copy of the plan
                new_plan = Plan()
                new_plan.observations = current_plan.observations[:]
                new_plan.add_observation(obs)
                # Copy of remaining obs
                new_remaining = obslist_deepcopy(remaining_observations)
                new_remaining.remove(obs)

                update_start_from_prev(new_remaining, obs)
                new_score = new_plan.evaluate_plan()
                PQ_next.put(self.PrioritizedItem(-new_score, new_plan, new_remaining))

            # If PQ_current is empty, move top-K from PQ_next to PQ_current
            if PQ_current.empty():
                print(f"Went through the {self.depth} level")
                # print(f"Current best plan: {best_current_plan}")
                self.depth += 1
                # Put top-K plans in the PQ_current queue
                best_score = 0.0
                for _ in range(min(K, PQ_next.qsize())):
                    pq_current_item = PQ_next.get()
                    # Update the best plan if this one is better
                    if pq_current_item.score < -best_score:
                        best_plan = pq_current_item.plan
                        best_score = pq_current_item.score
                    PQ_current.put(pq_current_item)
                # Update score of best plan

                # Clear PQ_next for the next iteration
                PQ_next = PriorityQueue()

        return best_plan


# Genetic algorithm scheduler
class GeneticAlgorithm:
    def __init__(self) -> None:
        self.total_counter: int = 0

    def nsga2(
        self,
        available_observations: List[Observation],
        max_plan_length: int,
        population_size: int,
        generations: int,
    ):
        """Non-dominated Sorting Genetic Algorithm II (NSGA-II) implementation"""
        return None
