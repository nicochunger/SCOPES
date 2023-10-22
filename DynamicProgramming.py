""" NOTE: Rough framework of dynamic programming that ChatGPT generated as an alternative
to the generateQ function from Rooyen et al. 2018. I think its similar to what they intended. """

# TODO: Change data type of Observation and Plan

from typing import List, Dict, Tuple

# Function to apply a start time to a single observation
def apply_start_time(observation: Dict, current_time: int) -> None:
    observation['startTime'] = current_time

# Function to check if an observation is feasible
def feasible(observation: Dict) -> bool:
    # Implement feasibility check logic here
    return True

# Function to evaluate the entire observation plan
def evaluateP(plan: List[Dict]) -> float:
    # Evaluate the whole observation plan
    total_score = sum(obs['score'] for obs in plan)
    return total_score / len(plan) if len(plan) > 0 else 0

# Function to update the start time after an observation has been added to the plan
def update_start_time(current_time: int, observation: Dict) -> int:
    # Implement time update logic here
    return observation['endTime']


# # Dynamic Programming table to store previously computed best scores and plans
# DP = {}

# # Core recursive function for dynamic programming approach
# def dp_recursion(remaining_observations: List[Dict], current_time: int, current_plan: List[Dict]) -> Tuple[float, List[Dict]]:
#     # Define the current state as a tuple
#     state = (tuple(obs['id'] for obs in remaining_observations), current_time)
    
#     # Check if state has already been computed
#     if state in DP:
#         return DP[state]
    
#     # Base case: No remaining observations, evaluate current plan
#     if len(remaining_observations) == 0:
#         score = evaluateP(current_plan)
#         DP[state] = (score, current_plan)
#         return score, current_plan
    
#     # Initialize variables to hold the best score and corresponding plan
#     best_score = float('-inf')
#     best_plan = []
    
#     # Loop through remaining observations to consider adding each to the plan
#     for obs in remaining_observations:
#         # Create a copy of remaining observations and remove the current one
#         remaining_copy = remaining_observations.copy()
#         remaining_copy.remove(obs)
        
#         # Apply the start time to the observation
#         apply_start_time(obs, current_time)
        
#         # Check if adding this observation is feasible
#         if feasible(obs):
#             # Update the current time based on the end time of the added observation
#             new_time = update_start_time(current_time, obs)
            
#             # Add observation to current plan to form new plan
#             new_plan = current_plan + [obs]
            
#             # Recursive call to find best plan from this point forward
#             _, temp_plan = dp_recursion(remaining_copy, new_time, new_plan)
            
#             # Evaluate this complete plan
#             score = evaluateP(temp_plan)
            
#             # Update best score and plan if this plan is better
#             if score > best_score:
#                 best_score = score
#                 best_plan = temp_plan
                
#     # Store the best score and plan for this state
#     DP[state] = (best_score, best_plan)
    
#     return best_score, best_plan

# # Main function to find optimal observation plan using dynamic programming
# def optimal_plan_dp(observations: List[Dict], start_time: int) -> Tuple[float, List[Dict]]:
#     return dp_recursion(observations, start_time, [])

# # Example usage
# observations = [
#     {'id': 1, 'startTime': 0, 'endTime': 2, 'score': 3},
#     {'id': 2, 'startTime': 0, 'endTime': 2, 'score': 4},
#     {'id': 3, 'startTime': 0, 'endTime': 2, 'score': 1},
# ]
# start_time = 0

# # Run the algorithm
# best_score, best_plan = optimal_plan_dp(observations, start_time)
# print("Best achievable score:", best_score)
# print("Best plan:", best_plan)
