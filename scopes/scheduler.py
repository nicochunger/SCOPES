from copy import deepcopy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, List, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from .merits import airmass_efficiency, end_time
from .scheduler_components import Merit, Night, Observation, Overheads, Plan


## ----- BASE SCHEDULER CLASS ----- ##
class Scheduler:
    """
    Base class for all schedulers.

    This class doesn't actually do any scheduling, but it sets up the necessary initializations
    and provides some helper functions that are common to all schedulers.
    """

    def __init__(
        self,
        night: Night,
        obs_list: List[Observation],
        overheads: Overheads,
        plan_start_time: float = None,
        plan_end_time: float = None,
    ) -> None:
        """
        Initializes the Scheduler class. Checks the validity of the plan start time and sets the
        start time of all observations. Also runs the skypath function for each observation.

        Parameters
        ----------
        night : Night
            The night object that defines the observable time.
        obs_list : List[Observation]
            The list of observations to schedule.
        overheads : Overheads
            The overheads object that defines the transition times between observations.
        plan_start_time : float, optional
            The start time of the plan in Julian Date. If None, it is set to the start of the
            observable night (as defined in the night object).
        plan_end_time : float, optional
            The end time of the plan in Julian Date. If None, it is set to the end of the
            observable night (as defined in the night object).
        """
        print("Preparing observations for scheduling...")
        # Set plan_start_time to the start of the observable night if it's None, otherwise use the given value
        self.plan_start_time = (
            night.obs_within_limits[0] if plan_start_time is None else plan_start_time
        )

        # Check that the selected plan start time is within the observable night limits
        if not (
            night.obs_within_limits[0]
            <= self.plan_start_time
            < night.obs_within_limits[1]
        ):
            raise ValueError(
                f"plan_start_time ({self.plan_start_time}) is outside the observable night limits "
                f"({night.obs_within_limits[0]} to {night.obs_within_limits[1]})."
            )
        # Set plan_end_time to the end of the observable night if it's None, otherwise use the given value
        custom_end_time = plan_end_time is not None
        self.plan_end_time = (
            night.obs_within_limits[1] if plan_end_time is None else plan_end_time
        )

        # Check that the selected plan end time is within the observable night limits
        if not (
            night.obs_within_limits[0]
            <= self.plan_end_time
            <= night.obs_within_limits[1]
        ):
            raise ValueError(
                f"plan_end_time ({self.plan_end_time}) is outside the observable night limits "
                f"({night.obs_within_limits[0]} to {night.obs_within_limits[1]})."
            )

        # Check that obs_list is a list of Observation objects
        if not isinstance(obs_list, list):
            raise TypeError("obs_list has to be a list of Observation objects.")
        # Check that obs_list is not empty and contains more than one observation
        if len(obs_list) <= 1:
            raise ValueError("obs_list has to contain more than one observation.")
        # Check that obs_list contains only Observation objects
        if not all(isinstance(obs, Observation) for obs in obs_list):
            raise TypeError("obs_list has to contain only Observation objects.")
        if not isinstance(night, Night):
            raise TypeError("night has to be a Night object.")
        if not isinstance(overheads, Overheads):
            raise TypeError("overheads has to be an Overheads object.")

        self.night = night
        self.obs_list = obs_list
        self.overheads = overheads
        # Set the start of all obs
        for obs in tqdm(self.obs_list, desc="Setting up observations"):
            if custom_end_time:
                obs.target.add_merit(
                    Merit("EndTime", end_time, "veto", {"time": self.plan_end_time})
                )
            # Set the night and start time
            obs.set_night(self.night)
            obs.set_start_time(self.plan_start_time)
            # Run skypath to calculate the path of the object during the night
            obs.skypath()
            obs.update_alt_airmass()

        # Calculate the extended time range for the culmination merit
        self.night.calculate_culmination_window(self.obs_list)

    def _check_max_plan_length(self, max_plan_length: Union[int, None]):
        """
        Checks the validity of the maximum plan length.

        Parameters
        ----------
        max_plan_length : int or None
            The maximum plan length. If None, it is set to the number of observations.

        Raises
        ------
        ValueError:
            If max_plan_length is greater than the number of observations or if it is not a
            positive integer or None.

        Returns
        -------
        max_plan_length : int
            The validated maximum plan length.
        """
        if max_plan_length is None:
            max_plan_length = len(self.obs_list)
        elif max_plan_length > len(self.obs_list):
            raise ValueError(
                "max_plan_length should be less than or equal to the number of observations."
            )
        if max_plan_length and max_plan_length <= 0:
            raise ValueError("max_plan_length should be a positive integer or None.")
        return max_plan_length

    def _obslist_deepcopy(self, obslist: List[Observation]):
        """
        An implementation of deepcopying a list of observations by creating new emtpy observations
        and assigning the attributes of the original observations to the new ones.

        This is a workaround for the fact that deepcopy is very slow for these types of objects.
        This alone can speed up the scheduling algorithms by a factor of 10.

        The only caveat is that this is not actually a deepcopy as the Target objects inside the
        Observation objects are not copied. This means is you make a modification on the Targets
        within these observation then this will affect the original observations as well.

        Parameters
        ----------
        obslist : List[Observation]
            The list of observations to deepcopy
        """
        new_obslist = []
        for obs in obslist:
            new_obs = Observation.__new__(Observation)
            new_obs.__dict__ = obs.__dict__.copy()
            new_obslist.append(new_obs)
        return new_obslist

    def _plan_deepcopy(self, plan: Plan):
        """
        An implementation of deepcopying a plan by creating new emtpy observations and assigning
        the attributes of the original observations to the new ones.

        This is a workaround for the fact that deepcopy is very slow for these types of objects.
        This alone can speed up the scheduling algorithms by a factor of 10.

        Parameters
        ----------
        plan : Plan
            The plan to deepcopy
        """
        # Create a new empty Plan
        new_plan = Plan()
        # Deepcopy the observations
        obs_copy = self._obslist_deepcopy(plan.observations)
        # Add the observations to the new plan
        for obs in obs_copy:
            new_plan.add_observation(obs)
        # Evaluate plan
        new_plan.evaluate_plan()
        return new_plan

    def update_start_times(
        self, observations: List[Observation], new_start_time: float
    ):
        """
        Update the start time of all observations in the list based on defined start time.

        Parameters
        ----------
        observations : List[Observation]
            The list of observations to update
        new_start_time : float
            The new start time to set for all observations
        """
        for obs in observations:
            obs.set_start_time(new_start_time)
            obs.update_alt_airmass()

    def update_start_from_prev(
        self, observations: List[Observation], previous_obs: Observation
    ):
        """
        Update the start time of all observations in the list based on the previous observation.

        Parameters
        ----------
        observations : List[Observation]
            The list of observations to update
        previous_obs : Observation
            The previous observation
        """
        for obs in observations:
            self.transition(previous_obs, obs)

    def transition(self, obs1: Observation, obs2: Observation):
        """
        Use the overheads class to calculate the transition time from obs1 to obs2. Then update
        the start time of obs2 and recalculate the score of obs2.

        Parameters
        ----------
        obs1 : Observation
            The first observation
        obs2 : Observation
            The second observation
        """
        total_overhead = self.overheads.calculate_transition(obs1, obs2)
        # Update the start time of obs2
        obs2.set_start_time(obs1.end_time + total_overhead)
        # Update the time array becaue the start time changed
        obs2.update_alt_airmass()
        # Calculate new rank score based on new start time
        obs2.feasible()
        obs2.evaluate_score()

    def move_observation(self, plan: Plan, index: int, new_index: int) -> Plan:
        """
        Move an observation to a new index in the plan.

        V2: Implement it more efficiently. Not as a series of swaps but directly placing the
        observation at the new location and moving all the observations in the middle as a block.

        Parameters
        ----------
        plan : Plan
            The plan to move the observation in
        index : int
            The index of the observation to move
        new_index : int
            The new index to move the observation to

        Returns
        -------
        Plan
            The plan with the moved observation
        """
        # Ensure indices are within the range of the plan's observations
        if index < 0 or index >= len(plan.observations):
            raise IndexError(f"index ({index}) is out of bounds.")
        if new_index < 0 or new_index >= len(plan.observations):
            raise IndexError(f"new_index {new_index} is out of bounds.")

        # If indices are the same there is no need to swap, return the original plan
        if index == new_index:
            return plan

        # Create a deep copy of the plan
        moved_plan = self._plan_deepcopy(plan)
        obss = moved_plan.observations

        # Move the observation to the new index
        moved_observation = obss.pop(index)
        obss.insert(new_index, moved_observation)

        left_idx = min(index, new_index)
        start_idx = left_idx - 1 if left_idx > 0 else 0
        # Update start time to end time of previous
        if start_idx == 0:
            # Beginning of plan, set start time to chosen start of plan
            obss[0].update_start_and_score(self.plan_start_time)
            obss[1].update_start_and_score(obss[0].end_time)
        else:
            obss[left_idx].update_start_and_score(obss[start_idx].end_time)
        # Check that observations didn't become unfeasible
        if obss[left_idx].score == 0.0:
            # If any of the scores are 0, the plan is infeasible, return the original plan
            return plan

        # Calculate new overheads for the transitions for following observations
        for idx in range(start_idx, len(obss) - 1):
            # Calculate new overhead
            # new_overhead = self.overheads.calculate_transition(obss[idx], obss[idx + 1])
            # # Update start time of next observation
            # obss[idx + 1].update_start_and_score(obss[idx].end_time + new_overhead)
            self.transition(obss[idx], obss[idx + 1])
            if obss[idx + 1].score == 0.0:
                # Observation becomes infeasible, abort and return the original plan
                return plan

        # Evaluate plan before returning
        moved_plan.evaluate_plan()

        return moved_plan

    def swap_observations(self, plan: Plan, index1: int, index2: int) -> Plan:
        """
        Swaps any two observations in the plan adjusts the start times and overheads for this
        new schedule.

        Parameters
        ----------
        plan : Plan
            The plan to swap observations in
        index1 : int
            The index of the first observation to swap
        index2 : int
            The index of the second observation to swap

        Returns
        -------
        Plan
            The plan with the swapped observations
        """
        # Ensure indices are within the range of the plan's observations
        if (
            index1 < 0
            or index1 >= len(plan.observations)
            or index2 < 0
            or index2 >= len(plan.observations)
        ):
            raise IndexError("Swap indices are out of bounds.")

        if index1 == index2:
            # No need to swap, return the original plan
            return plan

        # Create copy of the plan to not modify the original one
        plan_copy = self._plan_deepcopy(plan)

        # Swap the observations
        obss = plan_copy.observations
        obss[index1], obss[index2] = obss[index2], obss[index1]
        # Swap their start times and update the alt airmasses
        # Save the start time of obs1 before it gets changed
        obs_idx1_start = obss[index1].start_time
        obss[index1].update_start_and_score(obss[index2].start_time)
        obss[index2].update_start_and_score(obs_idx1_start)
        if not (obss[index2].score or obss[index1].score):
            # If any of the scores are 0, the plan is infeasible, return the original plan
            return plan

        # Calculate new overheads for the transitions between the previous and next observations
        # Calculate new transition for all obs from index1-1 until the end.
        left_idx = min(index1, index2)
        start_idx = left_idx - 1 if left_idx > 0 else 0
        for idx in range(start_idx, len(obss) - 1):
            # Calculate new overhead
            new_overhead = self.overheads.calculate_transition(obss[idx], obss[idx + 1])
            # Update start time of next observation
            obss[idx + 1].update_start_and_score(obss[idx].end_time + new_overhead)
            if obss[idx + 1].score == 0.0:
                # Observation becomes infeasible, abort and return the original plan
                return plan

        # Evaluate plan
        plan_copy.evaluate_plan()

        return plan_copy

    def _get_best(self, plans, metric, comparison_plan):
        """
        Get the best plan from a list of plans based on a given metric and comparison plan.

        Parameters
        ----------
        plan : Plan
            The plan to check
        metric : str
            The metric to use for comparison
        comparison_plan : Plan
            The plan to compare the other plans to

        Returns
        -------
        bool
            True if the plan is better, False otherwise
        """
        if metric == "overheads":
            overheads = [plan.overhead_time for plan in plans]
            best_plan = plans[np.argmin(overheads)]
            if best_plan.overhead_time < comparison_plan.overhead_time:
                return best_plan
        elif metric == "score":
            scores = [plan.score for plan in plans]
            best_plan = plans[np.argmax(scores)]
            if best_plan.score > comparison_plan.score:
                return best_plan
        else:
            raise ValueError("Invalid metric. Use 'overheads' or 'score'.")

        return comparison_plan

    def _get_metric(self, plan, metric):
        """
        Get the metric of a plan based on the given metric.

        Parameters
        ----------
        plan : Plan
            The plan to get the metric from
        metric : str
            The metric to get

        Returns
        -------
        float
            The value of the metric
        """
        if metric == "overheads":
            return plan.overhead_time
        elif metric == "score":
            return plan.score
        else:
            raise ValueError("Invalid metric. Use 'overheads' or 'score'.")

    def lsh_optimization(
        self,
        plan: Plan,
        method: str,
        metric: str,
        iterations: int = None,
        verbose: bool = False,
    ) -> Plan:
        """
        Optimize the plan using the LSH (Local Search Heuristic) algorithm. The algorithm works by
        going through all possible swaps or moves of observations and checking if the chosen metric
        is improved and takes the best one. If it is, the swap or move is accepted, otherwise it is
        rejected. The process is repeated for a number of iterations or until the metric doesn't
        improve. The algorithm can be used to optimize the plan based on the overhead time or the
        score of the plan. The method can be either "swap" or "move". Swap swaps two observations
        in the plan, while move moves an observation to a new spot in the plan.

        Parameters
        ----------
        plan : Plan
            The plan to optimize
        method : str
            The method to use for the optimization. Can be either "swap" or "move".
        metric : str
            The metric to use for the optimization. Can be either "overheads" or "score".
        iterations : int, optional
            The number of iterations to run the optimization for. If None, it runs until the score
            doesn't improve, by default None
        verbose : bool, optional
            If True, print the progress of the optimization, by default False

        Returns
        -------
        Plan
            The optimized plan
        """
        # Check that the method is either "swap" or "move" and assign the corresponding function
        if method not in ["swap", "move"]:
            raise ValueError("Method must be either 'swap' or 'move'.")
        if method == "move":
            operation_func = self.move_observation
        elif method == "swap":
            operation_func = self.swap_observations

        # Check that the metric is either "overheads" or "score"
        if metric not in ["overheads", "score"]:
            raise ValueError("Metric must be either 'overheads' or 'score'.")

        # Initialize the best plan to the original plan
        best_plan = self._plan_deepcopy(plan)
        if metric == "score":
            # Remove Culmination efficiency merit, and add the airmass efficiency merit
            for obs in best_plan.observations:
                # Check if the culmination_efficiency merit is present
                for merit in obs.target.efficiency_merits:
                    if "culmination" in merit.func.__name__.lower():
                        # Remove the culmination efficiency merit
                        obs.target.efficiency_merits.remove(merit)
                # Add the airmass efficiency merit
                obs.target.add_merit(
                    Merit(
                        "AirmassEfficiency",
                        airmass_efficiency,
                        merit_type="efficiency",
                    )
                )
                # Update scores with the new merit
                obs.feasible()
                obs.evaluate_score()
            # Evaluate the plan to recalculate score with new merits
            best_plan.evaluate_plan()

        if verbose:
            if metric == "overheads":
                print(f"Initial plan overhead time: {plan.overhead_time}")
            elif metric == "score":
                # Save initial scores and airmasses because of the change of merits
                intial_score = best_plan.score
                intial_avg_amass = best_plan.avg_airmass
                print(f"Initial plan score: {intial_score:.4f}")
                print(f"Initial avg airmass: {intial_avg_amass:.4f}")

        improved = True
        iteration_count = 0

        # TODO - Implement a tracking system to keep track of plans that were already rejected
        # This is to avoid recalculating the move operation of plans that were already done
        # rejected_states = {}

        # Make the forward and backward passes until the score doesn't improve
        while improved and (iterations is None or iteration_count < iterations):
            improved = False
            current_plan = self._plan_deepcopy(best_plan)
            # The indices of the observations to visit (variable U in the algorithm)
            pos_to_visit = list(range(len(plan.observations)))
            np.random.shuffle(pos_to_visit)  # Shuffles in-place
            move_count = 0
            for pos in pos_to_visit:
                # Go through all posible swaps and keep track of the overhead times
                plans = []
                for i in range(len(plan.observations)):
                    if i == pos:
                        # Skip the unnecesary swap with itself
                        continue
                    # Make the operation on the observation at pos
                    new_plan = operation_func(best_plan, pos, i)

                    if new_plan == best_plan:
                        # If the new plan is the same as the original plan, continue
                        # This happens when the swap is not feasible
                        continue
                    # Save the overhead time
                    plans.append(new_plan)

                # Find the plan with the lowest overhead time
                # Check that overheads is not empty
                if len(plans) == 0:
                    # If overheads is empty, there was no feasible swap, continue
                    continue
                else:
                    tmp_best_plan = self._get_best(plans, metric, best_plan)
                    # TODO idea: only accept the change if the time gain is "worth it"
                    # Where worth it has to be defined where we can look at the general score and if
                    # the score drop is too high to justify the time gain then the switch is not
                    # accepted. Or something like that...

                    if tmp_best_plan != best_plan:
                        # If the new plan is different from the original plan, accept the change
                        best_plan = tmp_best_plan
                        move_count += 1
                        improved = True

            # Update the iteration count
            iteration_count += 1

            if verbose:
                if metric == "overheads":
                    delta_t = current_plan.overhead_time - best_plan.overhead_time
                    print(f"\n----- Iteration {iteration_count} -----")
                    print(f"Overhead time: {best_plan.overhead_time}")
                    print(f"Time saved with {move_count} changes: {delta_t}\n")
                elif metric == "score":
                    delta_score = current_plan.score - best_plan.score
                    print(f"\n----- Iteration {iteration_count} -----")
                    print(f"Score: {best_plan.score:.4f}")
                    print(
                        f"Score improvement with {move_count} changes: {delta_score:.4f}\n"
                    )

        if verbose:
            print("Optimization finished.")
            if metric == "overheads":
                print(f"Optimized plan overhead time: {best_plan.overhead_time}")
                print(
                    "Total time saved: ", plan.overhead_time - best_plan.overhead_time
                )
            elif metric == "score":
                score_delta = intial_score - best_plan.score
                avg_amass_delta = intial_avg_amass - best_plan.avg_airmass
                print(
                    f"Optimized plan score: {best_plan.score:.4f} ({score_delta:+.4f})"
                )
                print(
                    f"Optimized plan avg airmass: {best_plan.avg_airmass:.4f} ({avg_amass_delta:+.4f})"
                )

        return best_plan


## ----- SPECIFIC SCHEDULERS ----- ##
class generateQ(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        # Call the parent class init
        super().__init__(*args, **kwargs)

    # Basic forward scheduler, greedy search
    def forwardP(
        self,
        start_obs: Union[Observation, float],
        available_obs: List[Observation],
        lookahead_distance=None,
    ):
        """
        Basic scheduler that simply continues a Plan from the starting observation by
        sequentially choosing the highest scoring observation.
        """

        # Set the lookahead distance to the number of available observations if not specified
        # Or to finish the night if there are more available observations than time in the night
        if (lookahead_distance is not None) and (
            len(available_obs) < lookahead_distance
        ):
            raise ValueError(
                f"Number of available observations ({len(available_obs)}) "
                f"must be more than or equal to lookahead distance ({lookahead_distance})"
            )
        elif lookahead_distance is None:
            # Equivalent to planning until the end of night or until no observations remain
            lookahead_distance = len(available_obs)

        # Initialize the Plan object according to if the starting codition is a Time or Observation
        observation_plan = Plan()
        if isinstance(start_obs, Observation):
            observation_plan.add_observation(start_obs)
            self.update_start_from_prev(available_obs, start_obs)
        elif isinstance(start_obs, float):
            self.update_start_times(available_obs, start_obs)

        else:
            raise TypeError(
                f"start_obs must be of type Observation or Time (as a float in jd), not {type(start_obs)}"
            )

        # Add candidate observation to plan K times

        for _ in range(lookahead_distance):
            # Initialize Q as an empty list to store ranked observations
            Q = []

            # Evaluate each available observation
            for o_prime in available_obs:
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
                available_obs.remove(o_double_prime)

                # Update the start time of all remaining observations
                self.update_start_from_prev(available_obs, o_double_prime)

        # Evaluate the plan before returning
        observation_plan.evaluate_plan()

        return observation_plan

    def run(self, max_plan_length=None, K: int = 5):
        """
        The run function for the generateQ Scheduler. The way it works is by using
        the forwardP function to generate a plan from the starting time to the end of the night
        but at each step it does this for the top K observations, and it chooses the observation
        that has the highest plan score at the end of the night. That would become the first
        observation of the plan. After each step it performs a Local Search Heuristic optimization
        that optimizes the order of the observation to minimize overheads.
        Then for the second it repeats the same process. It takes the top K observations runs
        forwardP until the end of the night, and assesses the plan score (but of the entire night,
        including the observation that was added before). It then chooses the observation that has
        the highest plan score at the end of the night, and that becomes the second observation of
        the plan. It repeats this process until the plan is full or it reaches the maximum plan
        length.

        At the end a final LSH optimization is run to maximize the score of the overall plan. This
        usually means reordering the observation to achieve a higher average airmass for the
        observations.

        Parameters
        ----------
        max_plan_length : int, optional
            The maximum length of the plan, by default None meaning it will go until the end of
            the night
        K : int, optional
            The number of top observations to consider at each step, by default 5

        Returns
        -------
        Plan
            The final plan
        """
        print("Creating the Plan...")
        # Check max_plan_length
        max_plan_length = self._check_max_plan_length(max_plan_length)

        # Create an empty plan to store the final results
        final_plan = Plan()

        # Create a deep copy of the available observations
        remaining_obs: List[Observation] = self._obslist_deepcopy(self.obs_list)
        # Iterate until the plan is full or remaining_obs is empty
        while remaining_obs and (len(final_plan) < max_plan_length):
            # Score each observation to sort them and pick the top K
            obs_scores = []
            for o in remaining_obs:
                if o.feasible():
                    score = o.evaluate_score()
                    obs_scores.append((score, o))
            if not obs_scores:
                # No feasible observations remain
                break
            obs_scores.sort(reverse=True, key=lambda x: x[0])

            top_k_observations = [obs for _, obs in obs_scores[:K]]

            # Track the best observation and corresponding plan
            best_observation = None
            best_plan = None

            for obs in top_k_observations:
                # Create a deep copy of available_obs
                remaining_obs_copy = self._obslist_deepcopy(remaining_obs)
                # Remove current obs from the copy
                remaining_obs_copy.remove(obs)

                if K == 1:
                    # Ensure the lookahead distance is at least 1
                    lookahead_distance = max_plan_length - len(final_plan) - 1
                else:
                    # Set the lookahead distance to the minimum of 5 and the remaining observations
                    lookahead_distance = min(5, max_plan_length - len(final_plan) - 1)
                # Generate plan using forwardP with the modified list
                plan = self.forwardP(
                    obs,
                    remaining_obs_copy,
                    lookahead_distance=lookahead_distance,
                )

                # If the current plan is better than the best, update best_plan and best_observation
                if not best_plan or (plan.score > best_plan.score):
                    best_plan = plan
                    best_observation = obs

            if K == 1:
                # If K=1, the best plan is the one generated by forwardP
                final_plan = best_plan
                break
            else:
                # Add the best observation to the final plan
                final_plan.add_observation(best_observation)
                # Optimize the overheads of the plan
                final_plan = self.lsh_optimization(
                    final_plan, method="move", metric="overheads"
                )
                # Remove the best observation from the available observations list
                remaining_obs.remove(best_observation)
                # self.update_start_from_prev(remaining_obs, best_observation)
                self.update_start_from_prev(remaining_obs, final_plan.observations[-1])

        # Perform a final score optimization
        final_plan = self.lsh_optimization(final_plan, method="move", metric="score")

        # Evaluate the final plan
        final_plan.evaluate_plan()
        print("Done!")

        return final_plan


# Dynamic programming scheduler using recursion
class DPPlanner(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        # Call the parent class init
        super().__init__(*args, **kwargs)
        self.DP = {}
        self.total_counter = 0
        self.saved_state_counter = 0

    def reset_dp(self):
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
        best_plan = Plan()

        # Evaluate feasibility and score of each observation
        for obs in remaining_observations:
            obs.feasible()
            obs.evaluate_score()
        # Select the top K observations to consider
        top_k_observations = sorted(
            remaining_observations, key=lambda x: x.score, reverse=True
        )[:K]

        # Loop through the top K observations to consider adding each to the plan
        for obs in top_k_observations:
            # Create a deep copy of current_plan first
            new_plan = deepcopy(current_plan)

            # Create a copy of the remaining observations
            remaining_copy = self._obslist_deepcopy(remaining_observations)
            # Remove the observation
            remaining_copy.remove(obs)

            # Check if adding this observation is feasible
            # NOTE: I think this check can be omitted as its already done in top_k_observations
            if obs.score > 0.0:
                # Add observation to the new plan
                new_plan.add_observation(obs)

                # Update the current time based on the end time of the added observation
                self.update_start_from_prev(remaining_copy, obs)

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

    def run(
        self,
        max_plan_length: int = None,
        K: int = 5,
    ) -> Plan:
        # Check max_plan_length
        max_plan_length = self._check_max_plan_length(max_plan_length)

        for obs in self.obs_list:
            obs.feasible()

        # Call the recursive function to find the best plan
        _, best_plan = self.dp_recursion(self.obs_list, Plan(), max_plan_length, K)

        return best_plan


# Beam search scheduler
class BeamSearchPlanner(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        # Call the parent class init
        super().__init__(*args, **kwargs)
        self.total_counter = 0
        self.depth = 0

    @dataclass(order=True)
    class PrioritizedItem:
        score: float
        plan: Any = field(compare=False)
        obs: Any = field(compare=False)

    def run(
        self,
        max_plan_length: int = None,
        K: int = 5,
    ) -> Plan:
        # Check max_plan_length
        max_plan_length = self._check_max_plan_length(max_plan_length)

        # Initialize two priority queues
        PQ_current: PriorityQueue = PriorityQueue()
        PQ_next: PriorityQueue = PriorityQueue()

        # Add initial state to the current priority queue
        PQ_current.put(self.PrioritizedItem(0, Plan(), self.obs_list))

        # Initialize best plan and best score to None and -inf
        best_plan: Plan = Plan()

        while not PQ_current.empty():
            self.total_counter += 1

            # Retrieve the highest-score plan from the current priority queue
            pq_current_item = PQ_current.get()
            # current_score = pq_current_item.score
            current_plan: Plan = pq_current_item.plan
            remaining_observations: List[Observation] = pq_current_item.obs

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
                new_remaining = self._obslist_deepcopy(remaining_observations)
                new_remaining.remove(obs)

                self.update_start_from_prev(new_remaining, obs)
                new_score = new_plan.evaluate_plan()
                PQ_next.put(self.PrioritizedItem(-new_score, new_plan, new_remaining))

            # If PQ_current is empty, move top-K from PQ_next to PQ_current
            if PQ_current.empty():
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
# class GeneticAlgorithm:
#     def __init__(self) -> None:
#         self.total_counter: int = 0

#     def nsga2(
#         self,
#         available_observations: List[Observation],
#         max_plan_length: int,
#         population_size: int,
#         generations: int,
#     ):
#         """Non-dominated Sorting Genetic Algorithm II (NSGA-II) implementation"""
#         return None
