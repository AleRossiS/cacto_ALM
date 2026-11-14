from .parameters import *
import pickle
from .network import ActorNetwork, load_actor 
import casadi as ca
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from .OCP_solver import OcpSingleIntegrator
import time
import os   # NOTE NEW
import functools
from environment_TO import SingleIntegrator_CAMS 
from TO import TO_JAX 



ENV = SingleIntegrator_CAMS('running_model', CONF_ALM)


def elliptical_constraints(x_next, ellips_params):
	"""
	Generate the mathematical constraint equations using CasADi.
	args:
		x_next: The next state of the system (position) for which we want to evaluate the constraints.
		ellips_params: A list of dictionaries, where each dictionary contains the parameters of an ellipse

	returns:
		A list of inequality expressions representing the constraints for each ellipse.  
	"""

	x_pos = x_next[0]
	y_pos = x_next[1]

	constraints = []
	for param in ellips_params:
		x_term = ((x_pos - param['x_ob'])**2) / (param['a']**2)
		y_term = ((y_pos - param['y_ob'])**2) / (param['b']**2)

		g_ellipse = x_term + y_term - 1
		constraints.append(g_ellipse)

	return constraints


# Batched rollout in JAX (scan over time, vmap over initial states)
@functools.partial(jax.jit, static_argnums=(0, 3))
def _rollout_batch(actor_apply, params, X0, N):
    """Roll out the actor from a batch of initial states X0 = [[x, y, t0], ...]."""
    def single(x0):
        def step(x, _):
            u = actor_apply(params, x)
            x_next = ENV.dynamics_tau_func_wrapped(x, u)   # x + dt*[u, 1]: time advances in seconds
            return x_next, (x_next, u)
        _, (X, U) = jax.lax.scan(step, x0, None, length=N)
        return jnp.vstack([x0, X]), U
    return jax.vmap(single)(X0)


def policy_rollout(actor, x_init_list, N):
    """
    Perform policy rollouts for a list of initial positions using the CACTO-JAX actor.

    Returns:
        all_x_traj: list of position trajectories, each of shape (N+1, 2).
        all_u_traj: list of control trajectories, each of shape (N, 2).
        all_costs:  list of total trajectory costs.
    """
    X0 = jnp.array([[p[0], p[1], 0.0] for p in x_init_list])   # network state [x, y, t = 0]
    X, U = _rollout_batch(actor.model.apply, actor.params, X0, N)
    X, U = np.array(X), np.array(U)

    all_x_traj = [X[i, :, :N_X] for i in range(len(X))]   # keep only the position (x, y) for plots/checks
    all_u_traj = [U[i] for i in range(len(U))]
    all_costs = [compute_trajectory_cost(X[i], U[i]) for i in range(len(X))]
    print('Rollout {}: {} trajectories'.format(actor.name, len(X)))

    return all_x_traj, all_u_traj, all_costs



def check_trajectory_feasibility(x_traj, obstacle_params):
	"""
	Check if a state trajectory satisfies all elliptical obstacle constraints
	and compute the constraint violation for each point in the trajectory.

	Args:
		x_traj: Trajectory array of shape (N+1, 2).
		obstacle_params: List of dictionaries containing obstacle parameters.

	Returns:
		is_feasible: True if no obstacle is intersected, False otherwise.
		point_violations: NumPy array of shape (N+1,) containing the violation depth at each point.
	"""
	n_points = len(x_traj)
	point_violations = np.zeros(n_points)

	for k, state in enumerate(x_traj):
		x_pos, y_pos = state[0], state[1]
		max_violation_at_k = 0.0

		for obs in obstacle_params:
			# Ellipse equation: (x - x_ob)^2 / a^2 + (y - y_ob)^2 / b^2 - 1 >= 0
			g_val = ((x_pos - obs['x_ob'])**2) / (obs['a']**2) + \
					((y_pos - obs['y_ob'])**2) / (obs['b']**2) - 1.0

			# Constraint violation occurs when g_val < 0
			if g_val < - TOLLERANCE:  # Use a small tolerance to avoid numerical issues
				violation = -g_val
				if violation > max_violation_at_k:
					max_violation_at_k = violation

		point_violations[k] = max_violation_at_k

	# Trajectory is feasible only if all point violations are below a small tolerance to avoid numerical issues
	is_feasible = np.all(point_violations <= TOLLERANCE)  

	# NOTE: the tollerance can be adjusted if needed

	return is_feasible, point_violations


  
def is_inside_obstacle(x_pos, y_pos, obstacle_params=OBSTACLE_PARAMETERS):
	"""
	Check if a given 2D point (x_pos, y_pos) lies inside any of the defined elliptical obstacles.

	Args:
		x_pos (float): X-coordinate of the point.
		y_pos (float): Y-coordinate of the point.
		obstacle_params (list): List of obstacle parameter dictionaries.

	Returns:
		bool: True if the point is strictly inside an obstacle, False otherwise.
	"""
	for obs in obstacle_params:
		g_val = ((x_pos - obs['x_ob'])**2) / (obs['a']**2) + \
				((y_pos - obs['y_ob'])**2) / (obs['b']**2) - 1.0
		if g_val < - TOLLERANCE:  # Use a small tolerance to avoid numerical issues
			return True
	return False


def generate_grid_initial_states(x_bounds=(-15.0, 15.0), y_bounds=(-15.0, 15.0), spacing=1.0, obstacle_params=OBSTACLE_PARAMETERS):
	"""
	Generate a uniform 2D grid of initial states within bounded limits based on step spacing,
	excluding points inside obstacles.

	Args:
		x_bounds (tuple): Min and max limits for X position.
		y_bounds (tuple): Min and max limits for Y position.
		spacing (float): Grid step size (resolution) between neighboring sample points.
		obstacle_params (list): List of obstacle parameter dictionaries.

	Returns:
		x_vals (ndarray): 1D array of sampled X coordinates.
		y_vals (ndarray): 1D array of sampled Y coordinates.
		grid_states (list): Valid initial state positions [x, y] outside obstacles.
	"""
	# Use spacing/2.0 tolerance on upper bound to include the endpoint
	x_vals = np.arange(x_bounds[0], x_bounds[1] + spacing / 2.0, spacing)
	y_vals = np.arange(y_bounds[0], y_bounds[1] + spacing / 2.0, spacing)
	
	grid_states = []
	for y in y_vals:
		for x in x_vals:
			if not is_inside_obstacle(x, y, obstacle_params):
				grid_states.append([x, y])
				
	return x_vals, y_vals, grid_states


# NOTE NEW: list of the actors saved by a run, identified by the number of updates (read from the NNs folder)
def list_checkpoints(n_try):
	"""
	Return the sorted list of the number of updates of the actors saved by run N_try
	(files actor_{n_update}.pkl in the NNs folder).
	"""
	folder = os.path.join(RESULTS_DIR, 'NNs', 'N_try_{}'.format(n_try))

	iters = []
	for f in os.listdir(folder):
		path = os.path.join(folder, f)
		# Empty files (run interrupted while saving) are skipped
		if f.startswith('actor_') and f.endswith('.pkl') and os.path.getsize(path) > 0:
			iters.append(int(f[len('actor_'):-len('.pkl')]))

	return sorted(iters)


# Trajectory cost = CACTO environment cost (same for both actors), computed as in TO.py
@jax.jit
def _traj_cost(X, U):
    U_pad = jnp.vstack([U, jnp.zeros((1, U.shape[1]))])   # terminal state with u = 0, as cost_arr in the TO
    return jnp.sum(jax.vmap(ENV.cost_tau_func_wrapped)(X, U_pad))


def compute_trajectory_cost(x_traj, u_traj):
    """
    Total cost of a trajectory.

    Args:
        x_traj: (N+1, 2) positions or (N+1, 3) full states [x, y, t].
        u_traj: (N, 2) controls.
    """
    x_traj = np.asarray(x_traj)
    if x_traj.shape[1] == N_X:   # positions only -> add the time column (t = k*dt)
        x_traj = np.hstack([x_traj, DT * np.arange(len(x_traj))[:, None]])
    return float(_traj_cost(jnp.asarray(x_traj), jnp.asarray(u_traj)))


def evaluate_grid_performance(x_vals, y_vals, grid_states, 
							  x_trajs_alm, u_trajs_alm, 
							  x_trajs_no_alm, u_trajs_no_alm, 
							  obstacle_params=OBSTACLE_PARAMETERS, 
							  target_pos=TARGET_POS):
	"""
	Evaluate trajectory feasibility and cumulative stage costs for all initial states across a 2D grid.

	Args:
		x_vals (ndarray): 1D array of sampled X coordinates.
		y_vals (ndarray): 1D array of sampled Y coordinates.
		grid_states (list): List of initial state positions [x, y].
		x_trajs_alm (list): Rollout state trajectories for ALM actor.
		u_trajs_alm (list): Control input trajectories for ALM actor.
		x_trajs_no_alm (list): Rollout state trajectories for NO ALM actor.
		u_trajs_no_alm (list): Control input trajectories for NO ALM actor.
		obstacle_params (list): List of obstacle parameter dictionaries.
		target_pos (list/ndarray): Target coordinates [x_g, y_g].

	Returns:
		feas_grid (ndarray): 2D array encoding feasibility status across the grid.
		cost_alm_grid (ndarray): 2D array of cumulative ALM rollout costs.
		cost_no_alm_grid (ndarray): 2D array of cumulative NO ALM rollout costs.
	"""
	# Preallocate 2D grid matrices (rows = Y, columns = X)
	feas_grid = np.full((len(y_vals), len(x_vals)), -1, dtype=int)
	cost_alm_grid = np.full((len(y_vals), len(x_vals)), np.nan)
	cost_no_alm_grid = np.full((len(y_vals), len(x_vals)), np.nan)

	# Create coordinate-to-index resolution lookup maps
	x_map = {np.round(val, 4): i for i, val in enumerate(x_vals)}
	y_map = {np.round(val, 4): i for i, val in enumerate(y_vals)}

	for idx, state in enumerate(grid_states):
		col = x_map[np.round(state[0], 4)]
		row = y_map[np.round(state[1], 4)]

		x_alm, u_alm = x_trajs_alm[idx], u_trajs_alm[idx]
		x_no_alm, u_no_alm = x_trajs_no_alm[idx], u_trajs_no_alm[idx]

		# Check feasibility
		feas_alm = x_alm is not None and check_trajectory_feasibility(x_alm, obstacle_params)[0]
		feas_no_alm = x_no_alm is not None and check_trajectory_feasibility(x_no_alm, obstacle_params)[0]

		# Encode status: 0: Both Fail | 1: ALM Fail & NO ALM OK | 2: ALM OK & NO ALM Fail | 3: Both OK
		if feas_alm and feas_no_alm:
			feas_grid[row, col] = 3
		elif feas_alm and not feas_no_alm:
			feas_grid[row, col] = 2
		elif not feas_alm and feas_no_alm:
			feas_grid[row, col] = 1
		else:
			feas_grid[row, col] = 0

		# Calculate costs
		cost_alm_grid[row, col] = compute_trajectory_cost(x_alm, u_alm) if x_alm is not None else np.nan
		cost_no_alm_grid[row, col] = compute_trajectory_cost(x_no_alm, u_no_alm) if x_no_alm is not None else np.nan

	return feas_grid, cost_alm_grid, cost_no_alm_grid



# Two TO solvers (same problem for both: CACTO dynamics, cost with w_ob = 0, obstacles as hard constraints)
TO_ILQR = TO_JAX(ENV, CONF_ALM)          # ILQR from official CACTO-JAX code (used in training)
OCP_IPOPT = OcpSingleIntegrator(ENV, N)  # IPOPT from CasADi (direct method, reference optimum)


def _solve_ilqr(X0, U_init):
    """Solve a batch of TO problems with AL-iLQR. Returns X, U, success flags, time per problem."""
    T = N * jnp.ones(X0.shape[0])
    solve = lambda: TO_ILQR.TO_System_Solve(X0, U_init, T, maxiter=CONF_ALM.maxiter, psd_delta=CONF_ALM.psd_delta)
    solve()[0].block_until_ready()   # first call compiles: excluded from the timing

    start = time.perf_counter()
    X, U, _, _, sf, _ = solve()
    X.block_until_ready()
    t = (time.perf_counter() - start) / X0.shape[0]   # batched solve: average time per problem
    return np.array(X), np.array(U), np.array(sf, dtype=bool), np.full(X0.shape[0], t)


def _solve_ipopt(X0, X_guess, U_guess):
    """Solve TO problems one by one with IPOPT. Returns X, U, success flags, time per problem."""
    n = X0.shape[0]
    X, U = np.zeros((n, N + 1, X0.shape[1])), np.zeros((n, N, N_U))
    sf, times = np.zeros(n, dtype=bool), np.zeros(n)
    for i in tqdm(range(n), desc='IPOPT', unit='traj'):
        start = time.perf_counter()
        try:
            X[i], U[i] = OCP_IPOPT.solve(np.array(X0[i]),
                                         None if X_guess is None else np.array(X_guess[i]),
                                         None if U_guess is None else np.array(U_guess[i]))
            sf[i] = compute_trajectory_cost(X[i], U[i]) < CONF_ALM.max_cost   # same cost check of the TO
        except RuntimeError:
            sf[i] = False   # IPOPT did not converge
        times[i] = time.perf_counter() - start
    return X, U, sf, times


def solve_TO_problems(actor, initial_states, N, solver=TO_SOLVER, fallback_on_fail=True):
    """
    Solve TO problems warm-started with the actor rollout.

    Args:
        actor: ActorNetwork used for the warm start.
        initial_states: list of initial positions [x, y].
        N: horizon.
        solver: 'ipopt' or 'ilqr'.
        fallback_on_fail: retry the failed problems with a naive warm start (zero control).

    Returns:
        x_opt_trajs (positions (N+1, 2) or None), u_opt_trajs, success_flags, cost_trajs, time_trajs,
        n_ws_fail (number of TO failed with the actor warm start, i.e. re-solved by the fallback).
    """
    X0 = jnp.array([[p[0], p[1], 0.0] for p in initial_states])
    X_ws, U_ws = _rollout_batch(actor.model.apply, actor.params, X0, N)   # warm start = actor rollout

    if solver == 'ilqr':
        X, U, sf, times = _solve_ilqr(X0, U_ws)
    elif solver == 'ipopt':
        X, U, sf, times = _solve_ipopt(X0, X_ws, U_ws)
    else:
        raise ValueError("solver must be 'ipopt' or 'ilqr'")

    # NOTE NEW: number of TO failed with the actor warm start (before the fallback) = quality of the actor as warm start
    n_ws_fail = int(np.sum(~sf))

    # Fallback: failed problems re-solved from zero control (as the stabilizing controller of CACTO)
    # This is done if the flag fallback_on_fail is True, otherwise the failed problems are returned as None
    if fallback_on_fail and not sf.all():
        idx = np.where(~sf)[0]
        U_zero = jnp.zeros((len(idx), N, N_U))
        if solver == 'ilqr':
            X2, U2, sf2, t2 = _solve_ilqr(X0[idx], U_zero)
        else:
            X2, U2, sf2, t2 = _solve_ipopt(X0[idx], None, U_zero)
        X[idx], U[idx], sf[idx], times[idx] = X2, U2, sf2, times[idx] + t2
        print('Fallback ({}): {}/{} recovered'.format(solver, sf2.sum(), len(idx)))

    x_opt_trajs = [X[i, :, :N_X] if sf[i] else None for i in range(len(sf))]
    u_opt_trajs = [U[i] if sf[i] else None for i in range(len(sf))]
    cost_trajs = [compute_trajectory_cost(X[i], U[i]) if sf[i] else np.nan for i in range(len(sf))]
    print('TO ({}) {}: success {}/{}'.format(solver, actor.name, sf.sum(), len(sf)))

    return x_opt_trajs, u_opt_trajs, list(sf), cost_trajs, list(times), n_ws_fail