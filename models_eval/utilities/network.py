import os
import pickle
from .parameters import RESULTS_DIR, load_conf   # importing parameters also adds the CACTO folder to sys.path
from NN_jax import ActorMLP


class ActorNetwork:
    """Wrapper of the CACTO-JAX actor: model + parameters + config of the run."""
    def __init__(self, name, conf, params):
        self.name = name
        self.conf = conf
        self.u_max = conf.u_max
        self.model = ActorMLP(conf)   # input normalization and tanh*u_max are inside the network
        self.params = params

    def __call__(self, x):
        """Control for a state x = [x, y, t] (or a batch of states)."""
        return self.model.apply(self.params, x)


def load_actor(n_try, n_update, name=None):
    """Load the actor saved by run N_try after n_update updates."""
    conf = load_conf(n_try)
    path = os.path.join(RESULTS_DIR, 'NNs', 'N_try_{}'.format(n_try), 'actor_{}.pkl'.format(n_update))
    with open(path, 'rb') as f:
        params = pickle.load(f)

    name = name if name is not None else 'N_try_{}_upd_{}'.format(n_try, n_update)
    actor = ActorNetwork(name, conf, params)
    stats = {'n_update': n_update}   # training statistics: filled later from the terminal log

    print('Actor loaded from {}'.format(path))
    return actor, stats