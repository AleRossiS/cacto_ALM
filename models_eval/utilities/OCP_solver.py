import casadi as ca
import numpy as np
from .parameters import N_U


# IPOPT Version, in the main i can chose to use IPOPT or ILQR

class OcpSingleIntegrator:
    """Optimal Control Problem of the CACTO single integrator, solved with IPOPT."""

    def __init__(self, env, N):
        self.env = env   # CACTO environment (CasADi functions for dynamics, cost and constraints)
        self.N = N

    def solve(self, x_init, X_guess=None, U_guess=None, max_iter=1000):
        """Solve the OCP from x_init = [x, y, t]. Returns X (N+1, 3) and U (N, 2)."""
        N = self.N
        opti = ca.Opti()
        x = opti.variable(len(x_init), N + 1)
        u = opti.variable(N_U, N)

        cost = 0
        for k in range(N):
            cost += self.env.cost_fun_tau(x[:, k], u[:, k])                         # running cost
            opti.subject_to(x[:, k + 1] == self.env.simulate_fun_tau(x[:, k], u[:, k]))   # dynamics
            opti.subject_to(self.env.constraint_fun(x[:, k + 1]) <= 0)                # obstacles (hard)
        cost += self.env.cost_fun_tau(x[:, N], ca.DM.zeros(N_U))   # terminal state with u = 0, as in the TO
        opti.subject_to(x[:, 0] == x_init)

        # Warm start (if no guess: constant state and zero control)
        opti.set_initial(x, X_guess.T if X_guess is not None else np.tile(x_init, (N + 1, 1)).T)
        opti.set_initial(u, U_guess.T if U_guess is not None else 0.0)

        opti.minimize(cost)
        opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': max_iter})
        sol = opti.solve()
        return np.array(sol.value(x)).T, np.array(sol.value(u)).T