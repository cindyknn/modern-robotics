import numpy as np
import copy
import sim
import goal
import rrt
import utils

########## TODO ##########

def plan_cost(plan):
    """
    Compute the cost of a motion plan based on the magnitude of 
    the controls. A lower cost generally corresponds to a shorter
    end-effector path.
    """
    cost = 0.0
    for node in plan:
        ctrl = node.get_control()
        if ctrl is None:
            continue
        # cost = velocity norm * duration
        cost += np.linalg.norm(ctrl[:3]) * ctrl[3]
    return cost


def optimize_traj(plan, pdef, iterations=200):
    """
    Perform trajectory optimization by refining node states.
    For each iteration, randomly select a node in the plan and re-propagate
    its associated control. If the resulting state is valid, update the node's
    state to the new state. The plan's cost is recalculated; if improved, the
    modification is kept. This iteratively refines the trajectory.
    """
    best = copy.deepcopy(plan)
    best_cost = plan_cost(best)
    
    for _ in range(iterations):
        cand = copy.deepcopy(best)
        idx = np.random.randint(len(cand))
        curr_node = cand[idx]
        ctrl = curr_node.get_control()
        
        if ctrl is None:
            continue
            
        # Re-propagate the control from this node
        new_state, valid = pdef.propagate(curr_node.state, ctrl)
        if valid and pdef.is_state_valid(new_state):
            # Update the node's state with the new valid state
            curr_node.state = new_state
            
            # Calculate the cost of this modified plan
            cost = plan_cost(cand)
            if cost < best_cost:
                best = cand
                best_cost = cost
    
    return best


##########################
