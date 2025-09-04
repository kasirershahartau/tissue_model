import numpy as np
from scipy.integrate import solve_ivp
from tyssue.solvers.viscous import EulerSolver, log


class IVPSolver(EulerSolver):
    def __init__( self,
        eptm,
        geom,
        model,
        history=None,
        auto_reconnect=False,
        manager=None,
        bounds=None,
        with_t1=False,
        with_t3=False,
    ):
        super().__init__(eptm, geom, model, history, auto_reconnect, manager, bounds, with_t1, with_t3)

    def clip_new_pos(self, old_pos, new_pos):
        change_in_pos = new_pos - old_pos
        change_in_pos = np.clip(change_in_pos, *self.bounds)
        return old_pos + change_in_pos


    def solve(self, tf, dt, on_topo_change=None, topo_change_args=(), method='RK45'):
        """Solves the system of differential equations from the current time
        to tf with steps of dt with scipy solve_IVP.

        Parameters
        ----------
        tf : float, final time when we stop solving
        dt : float, time step
        on_topo_change : function, optional, default None
             function of `self.eptm`
        topo_change_args : tuple, arguments passed to `on_topo_change`
        ode solving method to use (see scipy.integrate.solve_ivp documentation)
        """
        self.eptm.settings["dt"] = dt
        for t in np.arange(self.prev_t, tf + dt, dt):
            pos = self.current_pos
            new_pos = solve_ivp(self.ode_func, (0, dt), pos, t_eval=[dt]).y[:,0]
            if self.bounds is not None:
                new_pos = self.clip_new_pos(pos, new_pos)
            self.set_pos(new_pos)
            self.prev_t = t
            if self.manager is not None:
                self.manager.execute(self.eptm)
                self.geom.update_all(self.eptm)
                self.manager.update()

            if self.eptm.topo_changed:
                log.info("Topology changed")
                if on_topo_change is not None:
                    on_topo_change(*topo_change_args)
                self.eptm.topo_changed = False
            self.record(t)