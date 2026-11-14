import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # force JAX to use CPU (NOTE NEW: must be set before importing JAX)

import numpy as np
import utilities.functions as fnc
import utilities.plot_utilities as plt_util
from utilities.parameters import *
import matplotlib.pyplot as plt

"""
This script evaluates the performance of two policies (with and without ALM) over the actor training. It loads the saved models, performs rollouts, computes violation metrics, and visualizes the results. The evaluation is done on a set of predefined initial states and obstacle parameters.
The following metrics are computed and plotted:
- Rollout trajectories of both policies for each checkpoint from initial states (reference run of each method).
- Mean violation: mean (across all the points of all the rollouts) of the violation depth of the obstacle constraints.
- Number of violations: total number of violating points (all the rollouts).
- Cumulative violation severity: mean violation x number of violations.

The initial states used are INITIAL_STATES_HARD.

NOTE NEW: each method can have several runs (e.g. different seeds), listed in RUNS_ALM and RUNS_NO_ALM in utilities/parameters.py.
As for the TO cost (Fig. 4 of CACTO-BIC), every metric is computed for each run (one value per run, mean across initial conditions)
and plotted as the median across runs, with the shaded area = first and third quartiles across runs.
The checkpoints are the actors saved by the runs (one every save_interval updates), read automatically from the NNs folders.
At each checkpoint the statistics are computed over the runs that have that checkpoint: when all the runs of a method are finished, its curves stop.
The side-by-side rollouts are plotted for the reference run of each method (N_TRY_ALM, N_TRY_NO_ALM); a finished run shows its last actor.
"""

# Runs of each method and checkpoints saved by each run (number of updates), read from the NNs folders (NOTE NEW)
RUNS = {'ALM': RUNS_ALM, 'NO_ALM': RUNS_NO_ALM}
REF_RUN = {'ALM': N_TRY_ALM, 'NO_ALM': N_TRY_NO_ALM}   # reference run for the rollout plots
saved = {m: {r: fnc.list_checkpoints(r) for r in RUNS[m]} for m in RUNS}

# Common updates axis: all the checkpoints of all the runs
iters = sorted(set().union(*[set(c) for m in saved for c in saved[m].values()]))

for m in RUNS:
    for r in RUNS[m]:
        print(f"{m} run {r}: {len(saved[m][r])} checkpoints, last at {saved[m][r][-1]} updates")


def evaluate_policy(n_try, n_update, name):
    """Load the actor saved at n_update, perform the rollouts from INITIAL_STATES_HARD and compute the violation metrics. (NOTE NEW)"""
    actor, _ = fnc.load_actor(n_try, n_update, name=name)

    # --- Perform pure policy rollouts ---
    x_trajs, _, costs = fnc.policy_rollout(actor, INITIAL_STATES_HARD, N)

    # --- Compute Violation Metrics ---
    viols = []
    num_v = 0        # number of violating points (all the rollouts)
    for x_traj in x_trajs:
        _, p_viols = fnc.check_trajectory_feasibility(x_traj, OBSTACLE_PARAMETERS)
        viols.extend(p_viols)
        num_v += np.sum(p_viols > 0.0)

    return {'x': x_trajs, 'costs': costs,
            'mean_v': np.mean(viols),           # mean violation depth (all the points of all the rollouts)
            'num_v': num_v,                     # number of violating points
            'severity': np.mean(viols) * num_v} # cumulative violation severity (mean x count)


# Metrics computed for each run, then median and quartiles across runs (NOTE NEW)
METRICS = ['mean_v', 'num_v', 'severity']

# Initialize the results of each method (each method has its own updates list: the curves stop when all its runs are finished)
stats = {m: {'iters': []} for m in RUNS}
for m in RUNS:
    for k in METRICS:
        stats[m]['median_' + k], stats[m]['q1_' + k], stats[m]['q3_' + k] = [], [], []

rollouts_history = []
res_ref = {m: None for m in RUNS}   # results of the last evaluated actor of the reference run of each method
it_ref = {m: None for m in RUNS}


# Start the evaluation loop over the common updates axis
for i in iters:
    print(f"\n================================================")
    print(f"Evaluating policy rollouts at {i} updates")
    print(f"=================================================\n")

    for m in RUNS:
        values_runs = {k: [] for k in METRICS}   # one value per run

        # Evaluate every run of the method that has a checkpoint at i
        for r in RUNS[m]:
            if i not in saved[m][r]:
                continue
            res = evaluate_policy(r, i, f'{m}_run{r}')
            for k in METRICS:
                values_runs[k].append(res[k])
            print(f"• {i} updates - {m} run {r}: mean violation {res['mean_v']:.4f} | violating points {res['num_v']} | severity {res['severity']:.3f}")

            # The reference run is also used for the rollout plots
            if r == REF_RUN[m]:
                res_ref[m], it_ref[m] = res, i

        # Statistics across runs (as Fig. 4 of CACTO-BIC): median and quartiles of each metric
        if len(values_runs['mean_v']) > 0:
            stats[m]['iters'].append(i)
            for k in METRICS:
                stats[m]['median_' + k].append(np.median(values_runs[k]))
                stats[m]['q1_' + k].append(np.percentile(values_runs[k], 25))
                stats[m]['q3_' + k].append(np.percentile(values_runs[k], 75))
            print(f"  -> {m}: median mean violation {stats[m]['median_mean_v'][-1]:.4f} (Q1: {stats[m]['q1_mean_v'][-1]:.4f}, Q3: {stats[m]['q3_mean_v'][-1]:.4f}) over {len(values_runs['mean_v'])} run(s)\n")

    # --- Append current rollouts of the reference runs to history (a finished run keeps its last actor) ---
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


# --- Plot all rollouts in a single dynamic grid window (the grid size adapts to the number of checkpoints) ---
plt_util.plot_all_iterations_rollouts(
    rollouts_history=rollouts_history,
    obstacle_params=OBSTACLE_PARAMETERS,
    target_pos=TARGET_POS,
    show_cost=False,
    show_violations=True,
    save_name=f"policy_rollouts_converg_{RUN_TAG}",   # NOTE NEW: saved in images/
    title=f"Evolution of Policy Rollouts (runs ALM {N_TRY_ALM} / NO ALM {N_TRY_NO_ALM}) Across Training Updates"   # NOTE NEW
)

# --- Plot the convergence results (median and Q1-Q3 across runs, each method on its own updates list) ---
plt_util.plot_violation_convergence(
    stats_alm=stats['ALM'],
    stats_no_alm=stats['NO_ALM'],
    n_runs_alm=len(RUNS_ALM),
    n_runs_no_alm=len(RUNS_NO_ALM),
    save_name=f"policy_violations_converg_{RUN_TAG}"   # NOTE NEW: saved in images/
)

plt.show()  # Show all plots at the end of the evaluation loop
