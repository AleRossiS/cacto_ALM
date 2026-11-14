import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # force JAX to use CPU (NOTE NEW: must be set before importing JAX)

import numpy as np
import utilities.functions as fnc
import utilities.plot_utilities as plt_util
from utilities.parameters import *
import matplotlib.pyplot as plt

"""
Main script for evaluating the TO warm started with the actor policy, for the ALM and NO ALM cases. The script solves a set of TO problems (starting from INITIAL_STATES_HARD) for both policies across the actor training, compares the TO trajectories and plots the convergence of the TO cost and time.
The following metrics are computed and plotted:
- TO cost: as Fig. 4 of CACTO-BIC, median (across runs) of the mean cost (across initial conditions); shaded area = first and third quartiles across runs.
- TO time: as the cost, median (across runs) of the mean time per problem; shaded area = first and third quartiles across runs.
  (The quartiles are across runs, not across problems: with iLQR the problems are solved in batch, so every problem has the same average time.)

NOTE NEW: each method can have several runs (e.g. different seeds), listed in RUNS_ALM and RUNS_NO_ALM in utilities/parameters.py.
The checkpoints are the actors saved by the runs (one every save_interval updates), read automatically from the NNs folders.
At each checkpoint the statistics are computed over the runs that have that checkpoint: when all the runs of a method are finished, its curves stop.
The side-by-side TO trajectories are plotted for the reference run of each method (N_TRY_ALM, N_TRY_NO_ALM); a finished run shows its last actor.
The TO solver (TO_SOLVER: 'ipopt' or 'ilqr') is set in utilities/parameters.py. The same solver and the same TO problem are used for both actors.
"""

# Runs of each method and checkpoints saved by each run (number of updates), read from the NNs folders (NOTE NEW)
RUNS = {'ALM': RUNS_ALM, 'NO_ALM': RUNS_NO_ALM}
REF_RUN = {'ALM': N_TRY_ALM, 'NO_ALM': N_TRY_NO_ALM}   # reference run for the trajectory plots
saved = {m: {r: fnc.list_checkpoints(r) for r in RUNS[m]} for m in RUNS}

# Common updates axis: all the checkpoints of all the runs
iters = sorted(set().union(*[set(c) for m in saved for c in saved[m].values()]))

for m in RUNS:
	for r in RUNS[m]:
		print(f"{m} run {r}: {len(saved[m][r])} checkpoints, last at {saved[m][r][-1]} updates")
print(f"TO solver: {TO_SOLVER}")


def evaluate_TO(n_try, n_update, name):
	"""Load the actor saved at n_update, solve the TO problems from INITIAL_STATES_HARD warm-started with it and compute the mean cost and time. (NOTE NEW)"""
	actor, _ = fnc.load_actor(n_try, n_update, name=name)

	x_opt, _, flags, costs, times, n_ws_fail = fnc.solve_TO_problems(
			actor=actor,
			initial_states=INITIAL_STATES_HARD,
			N=N,
			solver=TO_SOLVER,
			fallback_on_fail=True
		)

	# Failed TO have cost NaN -> nan-aware mean (mean across initial conditions)
	costs = np.array(costs, dtype=float)
	return {'x': x_opt, 'costs': costs, 'success': int(np.sum(flags)),
			'mean_c': np.nanmean(costs), 'mean_t': np.mean(times),
			'ws_fail': 100.0 * n_ws_fail / len(INITIAL_STATES_HARD)}


# Initialize the results of each method (each method has its own updates list: the curves stop when all its runs are finished)
stats = {m: {'iters': [], 'median_c': [], 'q1_c': [], 'q3_c': [], 'median_t': [], 'q1_t': [], 'q3_t': [],
			 'median_f': [], 'q1_f': [], 'q3_f': []} for m in RUNS}   # NOTE NEW: f = % of warm-start failures

rollouts_history = []
res_ref = {m: None for m in RUNS}   # results of the last evaluated actor of the reference run of each method
it_ref = {m: None for m in RUNS}


# Start the evaluation loop over the common updates axis
for i in iters:
	print(f"\n================================================")
	print(f"--- Evaluating checkpoint at {i} updates ---")
	print(f"================================================\n")

	for m in RUNS:
		mean_costs_runs, mean_times_runs, ws_fail_runs = [], [], []   # one value per run (mean across initial conditions) (NOTE NEW: ws_fail_runs)

		# Evaluate every run of the method that has a checkpoint at i
		for r in RUNS[m]:
			if i not in saved[m][r]:
				continue
			res = evaluate_TO(r, i, f'{m}_run{r}')
			mean_costs_runs.append(res['mean_c'])
			mean_times_runs.append(res['mean_t'])
			ws_fail_runs.append(res['ws_fail'])   # NOTE NEW
			print(f"• {i} updates - {m} run {r}: mean cost {res['mean_c']:.2f} | mean time {res['mean_t']:.4f} | success {res['success']}/{len(INITIAL_STATES_HARD)} | warm-start failures {res['ws_fail']:.0f}%")

			# The reference run is also used for the trajectory plots
			if r == REF_RUN[m]:
				res_ref[m], it_ref[m] = res, i

		# Statistics across runs (as Fig. 4 of CACTO-BIC): median and quartiles of the mean cost
		if len(mean_costs_runs) > 0:
			stats[m]['iters'].append(i)
			stats[m]['median_c'].append(np.median(mean_costs_runs))
			stats[m]['q1_c'].append(np.percentile(mean_costs_runs, 25))
			stats[m]['q3_c'].append(np.percentile(mean_costs_runs, 75))
			stats[m]['median_t'].append(np.median(mean_times_runs))
			stats[m]['q1_t'].append(np.percentile(mean_times_runs, 25))   # NOTE NEW: quartiles across runs also for the time
			stats[m]['q3_t'].append(np.percentile(mean_times_runs, 75))
			stats[m]['median_f'].append(np.median(ws_fail_runs))   # NOTE NEW: warm-start failures, median and quartiles across runs
			stats[m]['q1_f'].append(np.percentile(ws_fail_runs, 25))
			stats[m]['q3_f'].append(np.percentile(ws_fail_runs, 75))
			print(f"  -> {m}: median of the mean cost {stats[m]['median_c'][-1]:.2f} (Q1: {stats[m]['q1_c'][-1]:.2f}, Q3: {stats[m]['q3_c'][-1]:.2f}) over {len(mean_costs_runs)} run(s)\n")

	# --- Append current TO trajectories of the reference runs to history (a finished run keeps its last actor) ---
	rollouts_history.append({
		'iter_alm': it_ref['ALM'],
		'iter_no_alm': it_ref['NO_ALM'],
		'alm_finished': i > saved['ALM'][N_TRY_ALM][-1],
		'no_alm_finished': i > saved['NO_ALM'][N_TRY_NO_ALM][-1],
		'x_alm': res_ref['ALM']['x'],
		'x_no_alm': res_ref['NO_ALM']['x'],
		'costs_alm': res_ref['ALM']['costs'],
		'costs_no_alm': res_ref['NO_ALM']['costs']
	})


# --- Plot all TO trajectories in a single dynamic grid window (the grid size adapts to the number of checkpoints) ---
plt_util.plot_all_iterations_rollouts(
    rollouts_history=rollouts_history,
    obstacle_params=OBSTACLE_PARAMETERS,
    target_pos=TARGET_POS,
    save_name=f"TO_{TO_SOLVER}_trajs_converg_{RUN_TAG}",   # NOTE NEW: saved in images/
    title=f"Evolution of TO Trajectories ({TO_SOLVER}, actor warm-start, runs ALM {N_TRY_ALM} / NO ALM {N_TRY_NO_ALM}) Across Training Updates"   # NOTE NEW
)

# --- Plot the convergence results (each method on its own updates list) ---
# NOTE NEW: cost = median across runs of the mean cost, band = Q1-Q3 across runs (Fig. 4 of CACTO-BIC).
#           time = same statistics as the cost (median and Q1-Q3 across runs of the mean time per problem).
#           warm-start failures = % of TO failed with the actor warm start (re-solved by the fallback), median and Q1-Q3 across runs.
#           The TO episodes solved in training are not available yet -> the TO episodes panel is commented in plot_cost_convergence.
plt_util.plot_cost_convergence(
    iters_alm=np.array(stats['ALM']['iters']),
    iters_no_alm=np.array(stats['NO_ALM']['iters']),
    mean_costs_alm=np.array(stats['ALM']['median_c']),
    q1_costs_alm=np.array(stats['ALM']['q1_c']),
    q3_costs_alm=np.array(stats['ALM']['q3_c']),
    mean_costs_no_alm=np.array(stats['NO_ALM']['median_c']),
    q1_costs_no_alm=np.array(stats['NO_ALM']['q1_c']),
    q3_costs_no_alm=np.array(stats['NO_ALM']['q3_c']),
    to_episodes_alm=None,
    to_episodes_no_alm=None,
    mean_times_alm=np.array(stats['ALM']['median_t']),
    q1_times_alm=np.array(stats['ALM']['q1_t']),
    q3_times_alm=np.array(stats['ALM']['q3_t']),
    mean_times_no_alm=np.array(stats['NO_ALM']['median_t']),
    q1_times_no_alm=np.array(stats['NO_ALM']['q1_t']),
    q3_times_no_alm=np.array(stats['NO_ALM']['q3_t']),
    ws_fail_alm=np.array(stats['ALM']['median_f']),   # NOTE NEW: warm-start failures panel
    q1_ws_fail_alm=np.array(stats['ALM']['q1_f']),
    q3_ws_fail_alm=np.array(stats['ALM']['q3_f']),
    ws_fail_no_alm=np.array(stats['NO_ALM']['median_f']),
    q1_ws_fail_no_alm=np.array(stats['NO_ALM']['q1_f']),
    q3_ws_fail_no_alm=np.array(stats['NO_ALM']['q3_f']),
    n_runs_alm=len(RUNS_ALM),
    n_runs_no_alm=len(RUNS_NO_ALM),
    save_name=f"TO_{TO_SOLVER}_cost_converg_{RUN_TAG}"   # NOTE NEW: saved in images/
)

plt.show()  # Show all plots at the end of the evaluation loop
