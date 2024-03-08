from copy import deepcopy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, List, Tuple, Union

from tqdm.auto import tqdm

from .scheduler_components import Night, Observation, Overheads, Plan


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
            The start time of the plan in Julian Date. If None, it is set to the start of the observable night.
        """
        print("Preparing observations for scheduling...")
        if plan_start_time is None:
            # If plan_start_time is None, set it to the start of the observable night
            self.plan_start_time = night.obs_within_limits[0]
        else:
            self.plan_start_time = plan_start_time
        # Check that the selected plan start time is within the night
        if self.plan_start_time < night.obs_within_limits[0]:
            raise ValueError(
                f"plan_start_time is before the start of the observable night ({night.obs_within_limits[0]})."
            )
        if self.plan_start_time >= night.obs_within_limits[1]:
            raise ValueError(
                f"plan_start_time is after the end of the observable night ({night.obs_within_limits[1]})."
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
        for obs in tqdm(self.obs_list):
            # Set the night and start time
            obs.set_night(self.night)
            obs.set_start_time(self.plan_start_time)
            # Run skypath to calculate the path of the object during the night
            obs.skypath()
            obs.update_alt_airmass()

        # Calculate the extended time range for the culmination merit
        self.night.calculate_culmination_window(self.obs_list)

    def check_max_plan_length(self, max_plan_length):
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

    def obslist_deepcopy(self, obslist):
        """
        An implementation of deepcopying a list of observations by creating new emtpy observations
        and assigning the attributes of the original observations to the new ones.

        This is a workaround for the fact that deepcopy is very slow for these types of objects.

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
        # Recalculate the score of obs2
        if obs2.end_time > self.night.obs_within_limits[1]:
            # Set score to 0 if observation goes beyond the end of the night
            obs2.score = 0.0
        else:
            # Update the time array becaue the start time changed
            obs2.update_alt_airmass()
            # Calculate new rank score based on new start time
            obs2.feasible()
            obs2.evaluate_score()


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
        The way it works is by using
        the forwardP function to generate a plan from the starting time to the end of the night
        but at each step it does this for the top K observations, and it chooses the observation
        that has the highest plan score at the end of the night. That would become the first
        observation of the plan. Then for the second it repeats the same process. It takes the
        top K observations runs forwardP until the end of the night, and assesses the plan score
        (but of the entire night, including the observation that was added before). It then chooses
        the observation that has the highest plan score at the end of the night, and that becomes
        the second observation of the plan. It repeats this process until the plan is full or it
        reaches the maximum plan length.

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
        max_plan_length = self.check_max_plan_length(max_plan_length)

        # Create an empty plan to store the final results
        final_plan = Plan()

        # Create a deep copy of the available observations
        remaining_obs = self.obslist_deepcopy(self.obs_list)
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
                remaining_obs_copy = self.obslist_deepcopy(remaining_obs)
                # Remove current obs from the copy
                remaining_obs_copy.remove(obs)

                # Generate plan using forwardP with the modified list
                plan = self.forwardP(
                    obs,
                    remaining_obs_copy,
                    lookahead_distance=max_plan_length - len(final_plan) - 1,
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
                # Remove the best observation from the available observations list
                remaining_obs.remove(best_observation)
                self.update_start_from_prev(remaining_obs, best_observation)

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
            remaining_copy = self.obslist_deepcopy(remaining_observations)
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
        max_plan_length = self.check_max_plan_length(max_plan_length)

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
        max_plan_length = self.check_max_plan_length(max_plan_length)

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
                new_remaining = self.obslist_deepcopy(remaining_observations)
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
