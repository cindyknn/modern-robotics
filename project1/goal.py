import numpy as np
import jac


class Goal(object):
    """
    A trivial goal that is always satisfied.
    """

    def __init__(self):
        pass

    def is_satisfied(self, state):
        """
        Determine if the query state satisfies this goal or not.
        """
        return True


class RelocateGoal(Goal):
    """
    The goal for relocating tasks.
    (i.e., pushing the target object into a circular goal region.)
    """

    def __init__(self, x_g=0.2, y_g=-0.2, r_g=0.1):
        """
        args: x_g: The x-coordinate of the center of the goal region.
              y_g: The y-coordinate of the center of the goal region.
              r_g: The radius of the goal region.
        """
        super(RelocateGoal, self).__init__()
        self.x_g, self.y_g, self.r_g = x_g, y_g, r_g

    def is_satisfied(self, state):
        """
        Check if the state satisfies the RelocateGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        """
        stateVec = state["stateVec"]
        x_tgt, y_tgt = stateVec[7], stateVec[8] # position of the target object
        if np.linalg.norm([x_tgt - self.x_g, y_tgt - self.y_g]) < self.r_g:
            return True
        else:
            return False


class GraspGoal(Goal):
    """
    The goal for grasping tasks.
    (i.e., approaching the end-effector to a pose that can grasp the target object.)
    """

    def __init__(self):
        super(GraspGoal, self).__init__()
        self.jac_solver = jac.JacSolver() # the jacobian solver

    def is_satisfied(self, state):
        """
        Check if the state satisfies the GraspGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        returns: True or False.
        """
        ########## TODO ##########
        
        state_target = state["stateVec"][7:10]
        pose_target = state["stateVec"][7:9]
        ee_coords, ee_orient = self.jac_solver.forward_kinematics(state["stateVec"])
        ee_coords[0] -= 0.4
        ee_coords[1] -= 0.2 
        ee_planarpos = ee_coords[:2]
        dist = np.linalg.norm(pose_target - ee_planarpos)
        w = ee_orient[3]
        
        # Angle
        ee_angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
        gamma = (ee_angle - state_target[2]) % 1.57079632      # % pi/2
        gamma = min(gamma, 1.57079632 - gamma)

        # Distances
        d1 = np.cos(gamma) * dist
        d2 = np.sin(gamma) * dist
        if (d1 > 0.01 or d2 > 0.02 or gamma > 0.2):
            return False
        else:
            return True


        ##########################
        