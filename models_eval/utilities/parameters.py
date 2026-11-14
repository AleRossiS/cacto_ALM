# NOTE NEW: parameters are read from the saved configs of the CACTO-JAX runs (no hard-coded values)
import os
import sys
import importlib.util
import numpy as np   # NOTE NEW

RANDOM_SEED = 13

# --- Paths ---
CACTO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # cacto_ALM folder
RESULTS_DIR = os.path.join(CACTO_ROOT, 'Results Single Integrator', 'Results set test - 1')
if CACTO_ROOT not in sys.path:
    sys.path.insert(0, CACTO_ROOT)   # to import NN_jax, environment_TO, TO from the CACTO code

# --- Runs to compare ---
# NOTE NEW: list of runs (e.g. different seeds) for each method. The convergence of the TO cost is plotted as in Fig. 4 of CACTO-BIC:
# median (across runs) of the mean cost (across initial conditions), shaded area = first and third quartiles across runs.
RUNS_ALM = [1, 2, 3, 4, 5]
RUNS_NO_ALM = [11, 12, 13, 14, 15]

# Reference run of each method: used by the single-run evaluations (main_*_eval.py) and for the rollout plots
N_TRY_ALM = RUNS_ALM[1]
N_TRY_NO_ALM = RUNS_NO_ALM[1]

def load_conf(n_try):
    """Load the configuration file saved by the run N_try."""
    path = os.path.join(RESULTS_DIR, 'Configs', 'conf_single_integrator_{}.py'.format(n_try))
    spec = importlib.util.spec_from_file_location('conf_{}'.format(n_try), path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    return conf

CONF_ALM = load_conf(N_TRY_ALM)
CONF_NO_ALM = load_conf(N_TRY_NO_ALM)
conf = CONF_ALM   # reference config for the problem geometry (checked below to be equal in both runs)


# Keep the same cost and constrained solver for both cases, so that the comparison is fair
CONF_ALM.CACTO_ALM = 1
CONF_ALM.cost_weights_running[3:6] = 0.0 

# --- Problem parameters ---
N = conf.NSTEPS               # horizon
DT = conf.dt                  # time step
N_X = conf.nx                 # position dimension (x, y); the network state is [x, y, t]
N_U = conf.nb_action          # control dimension
U_MAX = conf.u_max[0]         # control bound (tanh output of the actor)
TARGET_POS = [conf.x_des, conf.y_des]

# Obstacles: the config stores full widths/heights (A, B) -> semi-axes a = A/2, b = B/2
OBSTACLE_PARAMETERS = [
    {'x_ob': conf.XC1, 'y_ob': conf.YC1, 'a': conf.A1 / 2, 'b': conf.B1 / 2},   # vertical, back
    {'x_ob': conf.XC2, 'y_ob': conf.YC2, 'a': conf.A2 / 2, 'b': conf.B2 / 2},   # horizontal, top
    {'x_ob': conf.XC3, 'y_ob': conf.YC3, 'a': conf.A3 / 2, 'b': conf.B3 / 2},   # horizontal, bottom
]

# All the runs must have the same problem, otherwise the comparison is meaningless (NOTE NEW: checked for every run in the lists)
for n_try in RUNS_ALM + RUNS_NO_ALM:
    conf_run = load_conf(n_try)
    for name in ['NSTEPS', 'dt', 'x_des', 'y_des', 'obs_param', 'u_max']:
        assert np.all(np.asarray(getattr(conf, name)) == np.asarray(getattr(conf_run, name))), 'Run {} differs in {}'.format(n_try, name)

# NOTE NEW: tolerance for the feasibility check = the one of the TO (constraints_threshold of AL-iLQR).
# The violation is g = 1 - ellipse_equation(x) (same quantity used by the TO), so the TO solutions accepted by the solver
# (max violation < constraints_threshold) are feasible also here. 1e-2 means a point inside the ellipse by ~0.5% of the semi-axis (~1-3 cm).
TOLLERANCE = CONF_ALM.constraints_threshold

# NOTE NEW: tag with the two compared runs, used in the names of the saved figures
RUN_TAG = 'ALM{}_vs_NOALM{}'.format('-'.join(map(str, RUNS_ALM)), '-'.join(map(str, RUNS_NO_ALM)))

# --- Initial states (same as the ee_traj plots of the training) ---
INITIAL_STATES = [list(p) for p in conf.init_pos_sim]
INITIAL_STATES_HARD = [p for p in INITIAL_STATES if 1.0 < p[0] < 6.0 and abs(p[1]) < 3.0]   # inside the C

# NOTE NEW: TO solver used in the evaluation (same solver and same problem for both actors)
TO_SOLVER = 'ilqr'   # 'ipopt': reference optimum (direct method) | 'ilqr': AL-iLQR of CACTO (solver used in training)
# NOTE NEW: fixed axis limits of the trajectory plots (same in all panels, so that they can be compared). Default: sampling domain of the initial states
PLOT_XLIM = (conf.x_init_min[0], conf.x_init_max[0])
PLOT_YLIM = (conf.x_init_min[1], conf.x_init_max[1])
