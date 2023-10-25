from copy import deepcopy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from astropy.time import Time

from class_definitions import Observation, Plan


def get_observation_night(observation: Observation) -> Time:
    """Return the night of an observation."""
    sunset = observation.observer.sun_set_time(observation.start_time, which="previous")
    return sunset.date()


def update_start_times(observations: List[Observation], previous_obs: Observation):
    """Update the start time of all observations in the list based on the previous observation."""
    for obs in observations:
        obs.update_start_time(previous_obs)


## ----- SCHEDULERS ----- ##


# Basic forward scheduler, greedy search
def forwardP(
    start_obs: Union[Observation, Time],
    available_observations: List[Observation],
    lookahead_distance: int = None,
):
    """Basic scheduler that simply continues a Plan from the starting observation by
    sequentially choosing the highest scoring observation."""

    # Set the lookahead distance to the number of available observations if not specified
    # Or to finish the night if there are more available observations than time in the night
    if (lookahead_distance is not None) and (
        len(available_observations) <= lookahead_distance
    ):
        raise ValueError(
            f"Number of available observations ({len(available_observations)}) "
            f"must be less than or equal to lookahead distance ({lookahead_distance})"
        )
    elif lookahead_distance is None:
        lookahead_distance = len(available_observations)

    # Create deep copy of available observations
    Obs_copy = deepcopy(available_observations)

    # Initialize the Plan object according to if the starting codition is a Time or Observation
    observation_plan = Plan()
    if isinstance(start_obs, Observation):
        observation_plan.add_observation(start_obs)
        update_start_times(Obs_copy, start_obs)
    elif isinstance(start_obs, Time):
        for obs in Obs_copy:
            obs.start_time = start_obs
            obs.update_time_array()

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
            update_start_times(Obs_copy, o_double_prime)

    # Evaluate the plan before returning
    observation_plan.evaluate_plan()

    return observation_plan


# Dynamic programming scheduler using recursion
class DPPlanner:
    def __init__(self):
        self.DP: Dict = {}
        self.total_counter: int = 0
        self.saved_state_counter: int = 0

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
                update_start_times(remaining_copy, obs)

                # Recursive call to find best plan from this point forward
                _, temp_plan = self.dp_recursion(
                    remaining_copy, new_plan, max_plan_length, K
                )

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
    def __init__(self):
        self.total_counter: int = 0
        self.depth: int = 0

    @dataclass(order=True)
    class PrioritizedItem:
        score: float
        plan: Any = field(compare=False)
        obs: Any = field(compare=False)

    def dp_beam_search(
        self, initial_observations: List[Observation], max_plan_length: int, K: int = 5
    ) -> Plan:
        # Initialize two priority queues
        PQ_current: PriorityQueue = PriorityQueue()
        PQ_next: PriorityQueue = PriorityQueue()

        # Add initial state to the current priority queue
        PQ_current.put(self.PrioritizedItem(0, Plan(), initial_observations))

        # Initialize best plan and best score to None and -inf
        best_plan: Plan = Plan()
        # best_score = float('inf')

        while not PQ_current.empty():
            self.total_counter += 1

            # Retrieve the highest-score plan from the current priority queue
            pq_current_item = PQ_current.get()
            current_score = pq_current_item.score
            current_plan = pq_current_item.plan
            remaining_observations = pq_current_item.obs

            # Update the best plan if this one is better
            # print(f"Depth: {self.depth}   Current score: {current_score:.3f}")

            best_current_plan = Plan()
            if current_score < -best_current_plan.score:
                best_current_plan = current_plan

            # Check stopping criteria
            if len(current_plan) >= max_plan_length:
                best_plan = PQ_current.get().plan
                break

            Q = []
            # Generate child plans by extending the current plan with feasible observations
            for obs in remaining_observations:
                if obs.feasible():
                    score = obs.evaluate_score()
                    Q.append((score, obs))

            # Sort Q by score
            Q.sort(reverse=True, key=lambda x: x[0])
            for _, obs in Q[: K + 5]:
                new_plan = deepcopy(current_plan).add_observation(
                    obs
                )  # Assuming add_observation returns a modified plan
                new_remaining = deepcopy(remaining_observations)
                new_remaining.remove(obs)
                update_start_times(new_remaining, obs)
                new_score = (
                    new_plan.evaluate_plan()
                )  # Assuming evaluate_plan returns a score
                PQ_next.put(self.PrioritizedItem(-new_score, new_plan, new_remaining))

            # If PQ_current is empty, move top-K from PQ_next to PQ_current
            if PQ_current.empty():
                print(f"Went through the {self.depth} level")
                # print(f"Current best plan: {best_current_plan}")
                self.depth += 1
                # Put top-K plans in the PQ_current queue
                for _ in range(min(K, PQ_next.qsize())):
                    PQ_current.put(PQ_next.get())

                # print(f"PQ_current size: {PQ_current.qsize()}")
                # assert PQ_current.qsize() == min(K, PQ_next.qsize())

                # Clear PQ_next for the next iteration
                PQ_next = PriorityQueue()

        return best_plan
