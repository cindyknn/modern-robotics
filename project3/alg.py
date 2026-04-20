import numpy as np
import sim
import utils
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


OBS_CENTER = np.array([[0.9, 0], [0.25, 0.5], [-0.3, 0.5], [-1, 0.1], [0.3, -0.8]])
OBS_RADIUS = np.array([0.5, 0.3, 0.2, 0.5, 0.4])
SPH_RADIUS = 0.02

FK_Solver = utils.FKSolver() # forward kinematics solver


########## Task 1: Particle Weights Calculation ##########

def dist_to_closest_obs(x, y):
    """
    Find the distance between the stick's sphere end centered at (x, y) 
    to its closest cylinder obstacle.
    args:       x: The x-coordinate of the spherical end.
                y: The y-coordinate of the spherical end. 
    returns: dist: The distance to the closest obstacle.
    """
    ########## TODO ##########
    # dist = 0.0

    # Find center-to-center 2D distances from sphere to the 5 cylinders
    x_diff = x - OBS_CENTER[:, 0]
    y_diff = y - OBS_CENTER[:, 1]
    c_dists = np.sqrt(x_diff**2 + y_diff**2)

    # Surface-to-surface 2D distance = center distance - cylinder radius - sphere radius
    s_dists = c_dists - SPH_RADIUS - OBS_RADIUS

    # Closest cylinder has min |surface distance|
    closest_cyl = np.argmin(np.abs(s_dists))
    dist = s_dists[closest_cyl]

    ##########################
    return dist

def cal_weights(particles, obv, sigma=0.05):
    """
    Calculate the weights for particles based on the given observation.
    args:  particles: The particles represented by their states
                      Type: numpy.ndarray of shape (# of particles, 3)
                 obv: The given observation (the robot's joint angles). 
                      Type: numpy.ndarray of shape (7,)
               sigma: The standard deviation of the Gaussian distribution
                      for calculating likelihood (default: 0.05).
    returns: weights: The weights of all particles.
                      Type: numpy.ndarray of shape (# of particles,)
    """
    ########## TODO ##########
    # weights = None

    # Find sphere end position wrt robot base frame from joint angles
    x_robot, y_robot = FK_Solver.forward_kinematics_2d(obv)

    # Transform sphere position to world frame:
    # [x_world]   [cos θ  -sin θ] [x_robot]   [xi]
    # [y_world] = [sin θ   cos θ] [y_robot] + [yi]
    xi = particles[:, 0]
    yi = particles[:, 1]
    thetai = particles[:, 2]
    x_world = (x_robot * np.cos(thetai)) - (y_robot * np.sin(thetai)) + xi
    y_world = (x_robot * np.sin(thetai)) + (y_robot * np.cos(thetai)) + yi

    # Find distance from sphere to closest obstacle
    dists = np.array([dist_to_closest_obs(x_world[i], y_world[i]) for i in range(len(particles))])

    # Find weights using zero-mean Gaussian PDF and normalize so sum to 1
    unnorm_weights = scipy.stats.norm.pdf(dists, loc=0.0, scale=sigma)
    total = np.sum(unnorm_weights)
    weights = unnorm_weights / total

    ##########################
    return weights

def most_likely_particle(particles, obv):
    """
    Find the most likely particle.
    args:  particles: The particles represented by their states
                      Type: numpy.ndarray of shape (# of particles, 3)
                 obv: The given observation (the robot's joint angles). 
                      Type: numpy.ndarray of shape (7,)
    returns:     idx: The index of the most likely particle
                      Type: int
    """
    ########## TODO ##########
    # idx = 0

    # Find weights and return the particle of max weight
    weights = cal_weights(particles, obv)
    idx = int(np.argmax(weights))

    ##########################
    return idx


########## Task 2: Particle Filter ##########

def particle_filter(panda_sim, obvs, num_particles, sigma=0.05, delta=0.01, plot=True):
    """
    The Particle Filtering algorithm. 
    args:    panda_sim: The instance of the simulation. 
                        Type: sim.PandaSim (provided)
                  obvs: The given observations (the robot's joint angles)
                        Type: numpy.ndarray of shape (# of observations, 7)
         num_particles: The number of particles. 
                        Type: int
                 sigma: The standard deviation of the Gaussian distribution
                        for calculating likelihood (default: 0.05).
                 delta: The scale of the Gaussian for perturbing particles.
                        (default: 0.01)
                  plot: Whether to enable particle plot (True or False).
    returns:       est: The estimate of the pose of the robot base. 
                        Type: numpy.ndarray of shape (3,)
    """
    # initialize the particles and weights 
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    if plot:
        ax = utils.config_plot_ax()
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)

    for obv in obvs:
        panda_sim.set_joint_values(obv)

        ########## TODO ##########
        
        # Update weights
        weights = cal_weights(particles, obv, sigma)

        # Resample particles based on their weights
        idxs = np.random.choice(num_particles, size=num_particles, replace=True, p=weights)
        particles = particles[idxs]

        # Add noise to prevent particle collapse
        noise = np.random.normal(loc=0.0, scale=delta, size=particles.shape)
        particles = particles + noise

        if plot:
            plt.pause(0.01)      # Added because the 0.01s pause isn't enough time to render 3D scatter plot on my computer

        ##########################

        # plot the particles in the visualization
        if plot:
            utils.plot_pf(ax, particles, panda_sim.loc)
            plt.pause(0.01)
    est = particles.mean(0)
    return est


########## Task 3: Generate Observations Online ##########

def get_one_obv(panda_sim):
    """
    Control the robot in simulation to obtain an observation.
    args: panda_sim: The instance of the simulation. 
                     Type: sim.PandaSim (provided)
    returns:    obv: One observation found by this function.
                     Type: numpy.ndarray of shape (7,)
    """
    ########## TODO ##########
    obv = None
    
    # Save starting state
    state = panda_sim.save_state()

    while True:
        # Restore to saved state
        panda_sim.restore_state(state)

        # Pick random 3D translational direction
        dir = np.random.uniform(-1, 1, size=3)
        dir = dir / np.linalg.norm(dir)

        # Build 6D Cartesian velocity
        v = np.zeros(6)
        v[0:3] = dir * 0.5

        # Move in this direction
        for _ in range(500):
            if panda_sim.is_touch():
                # Safe: Robot end touched a cylinder
                jpos, _, _ = panda_sim.get_joint_states()
                obv = np.array(jpos[0:7])
                panda_sim.restore_state(state)
                return obv
            if panda_sim.is_collision():
                # Unsafe: robot body hit something
                break
            panda_sim.execute(v)

    ##########################
    return obv

def particle_filter_online(panda_sim, num_particles, sigma=0.05, delta=0.01, plot=True):
    """
    The online Particle Filtering algorithm. 
    args:     panda_sim: The instance of the simulation. 
                         Type: sim.PandaSim (provided)
          num_particles: The number of particles. 
                         Type: int
                  sigma: The standard deviation of the Gaussian distribution
                         for calculating likelihood (default: 0.05).
                  delta: The scale of the Gaussian for perturbing particles.
                         (default: 0.01)
                   plot: Whether to enable particle plot (True or False).
    returns:        est: The estimate of the pose of the robot base. 
                         Type: numpy.ndarray of shape (3,)
    """
    # initialize the particles and weights 
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    if plot:
        ax = utils.config_plot_ax()
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)

    ########## TODO ##########
    
    num_iters = 100 # The number of iterations. Feel free to change the value
    for _ in range(num_iters):        
        
        # Collect an observation online
        obv = get_one_obv(panda_sim)

        # Update weights
        weights = cal_weights(particles, obv, sigma=sigma)

        # Resample particles
        idxs = np.random.choice(num_particles, size=num_particles, replace=True, p=weights)
        particles = particles[idxs]

        # Add noise
        noise = np.random.normal(loc=0.0, scale=delta, size=particles.shape)
        particles = particles + noise

        # plot the particles in the visualization
        if plot: 
            utils.plot_pf(ax, particles, panda_sim.loc)
            plt.pause(0.18)     # Added because the 0.01s pause isn't enough time to render 3D scatter plot on my computer
    ##########################
    est = particles.mean(0)
    return est
