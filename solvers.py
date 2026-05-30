import numpy as np
from scipy.integrate import solve_ivp
from tyssue.solvers.viscous import EulerSolver, log
from tqdm import tqdm

class IVPSolver(EulerSolver):
    def __init__(self,
        inner,
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
        self.inner = inner

    def clip_new_pos(self, old_pos, new_pos):
        change_in_pos = new_pos - old_pos
        change_in_pos = np.clip(change_in_pos, *self.bounds)
        return old_pos + change_in_pos

    def _refresh_active_verts(self):
        """Recompute ``eptm.active_verts`` from the current ``is_active``
        column. Required after any topology event that drops or
        renumbers vertices (``reset_index`` invalidates the old
        positional indices stored in ``active_verts``)."""
        if "is_active" in self.eptm.vert_df.columns:
            self.eptm.active_verts = np.where(
                self.eptm.vert_df["is_active"].values
            )[0]

    def _record_at(self, t, dt):
        """Write a history snapshot at simulation time ``t`` and stamp
        the dt that produced the step. The stamp is stored as a column
        on ``face_df`` so it travels through every History backend
        without changes to the archive schema."""
        if self.history is None:
            return
        self.eptm.face_df["step_dt"] = float(dt)
        self.history.record(time_stamp=t)

    def solve(self, tf, dt, on_topo_change=None, topo_change_args=(),
              method='RK45', quasi_static=False, quasi_static_threshold=0.01,
              max_displacement=None, max_disp_factor=0.25,
              dt_min_factor=0.001, dt_increase_factor=1.1,
              save_interval=None,
              until_steady_state=False,
              lateral_inhibition_threshold=0.0,
              check_mechanical_steady=True,
              check_lateral_inhibition_steady=True):
        """Solves the ODE from the current time to tf with ADAPTIVE dt
        and edge-crossing safety nets.

        Parameters
        ----------
        tf : float
            final simulation time. When ``until_steady_state`` is True
            this becomes a safety CAP — the solver will stop earlier
            once the steady-state criteria are met.
        dt : float
            INITIAL time step. The solver may shrink dt when a step
            would move a vertex too far (and slowly ratchet it back
            toward this value on subsequent successful steps).
        max_displacement : float, optional
            Hard cap on the per-step maximum vertex displacement. If
            None (default), derived from ``max_disp_factor`` times
            ``eptm.minimal_bond_length`` so a vertex can't move more
            than a fraction of the shortest edge in one step.
        max_disp_factor : float, default 0.25
            Used to derive ``max_displacement`` when none is given.
        dt_min_factor : float, default 0.001
            Smallest dt the solver will try, expressed as a fraction
            of the initial dt. If a step is rejected and the next dt
            would fall below this floor, ``RuntimeError`` is raised.
        dt_increase_factor : float, default 1.1
            After every successful step, multiply dt by this factor
            (capped at the initial dt) to recover the original cadence
            once the dynamics calm down.
        save_interval : float, optional
            Wall-clock simulation-time interval between history
            snapshots. Defaults to the history's ``save_every`` if set,
            otherwise to the initial dt. Snapshots happen at constant
            time intervals regardless of iteration count.
        until_steady_state : bool, default False
            When True, stop the simulation as soon as the enabled
            steady-state criteria are all met (positions and/or
            lateral-inhibition levels barely change between iterations).
            ``tf`` becomes a wall-clock cap on the maximum simulation
            time.
        lateral_inhibition_threshold : float, default 0.0
            Per-step maximum allowed change in ``notch_level``,
            ``delta_level`` and ``repressor_level`` before the
            lateral-inhibition system is considered steady. Only used
            when ``until_steady_state`` AND
            ``check_lateral_inhibition_steady`` are True. The check is
            ``max(|new - old|) < threshold`` taken over all three
            level columns (whichever exist on ``face_df``).
        check_mechanical_steady : bool, default True
            Whether the mechanical steady-state criterion (positions
            change less than ``quasi_static_threshold``) participates
            in the steady-state decision. Set to False when the caller
            wants to halt purely on lateral-inhibition convergence.
        check_lateral_inhibition_steady : bool, default True
            Whether the lateral-inhibition convergence criterion
            participates in the decision. Set to False when the caller
            wants to halt purely on mechanical equilibrium.

        Rejection criteria (B + C in the discussion):
          - B (displacement): if max(|new_pos - pos|) > max_displacement
            the step is rejected without being applied; dt is halved.
          - C (negative area): after the candidate position is applied,
            if any face has a negative signed area (edge crossing
            flipped its orientation) the positions are reverted; dt is
            halved.

        Steady-state detection
        ----------------------
        When ``until_steady_state`` is True, after every ACCEPTED step
        we evaluate:

          - mech_ok = ``max(|new_pos - old_pos|) < quasi_static_threshold``
                     (only when ``check_mechanical_steady``)
          - li_ok   = ``max(|new_levels - old_levels|) < lateral_inhibition_threshold``
                     (only when ``check_lateral_inhibition_steady``)

        Both default to ``True`` when their flag is False (i.e. they
        don't gate the decision). Whichever criteria are enabled must
        all hold to stop. A topology change in the same step always
        forces ``li_ok = False`` (face counts can't be compared
        meaningfully across divisions / delaminations).
        """
        initial_dt = float(dt)
        dt = initial_dt
        dt_min = initial_dt * dt_min_factor

        # Default save interval: match the History's existing save_every
        # if set, otherwise save every initial dt (one snapshot per
        # "nominal" time step).
        if save_interval is None:
            inner_save_every = getattr(self.history, "save_every", None)
            save_interval = float(inner_save_every) if inner_save_every else initial_dt
        # Disable History's internal index-based throttle so OUR
        # constant-time scheduler controls when to record.
        if self.history is not None and hasattr(self.history, "save_every"):
            self.history.save_every = None

        # Derive the per-step displacement cap from the shortest edge.
        if max_displacement is None:
            min_bond = getattr(self.eptm, "minimal_bond_length", None)
            if min_bond is None or min_bond <= 0:
                min_bond = 0.05  # fallback
            max_displacement = max_disp_factor * float(min_bond)

        self.eptm.settings["dt"] = dt
        self.eptm.settings["initial_dt"] = initial_dt
        last_differentiation_t = -dt  # only for quasi-static mode
        self._refresh_active_verts()

        # Which lateral-inhibition columns exist on this sheet? Only
        # populated when the caller asked for an LI steady-state check.
        # Sampling once here (before the loop) is fine because these
        # columns aren't dropped mid-simulation — divisions only ADD
        # rows, delaminations REMOVE them, neither drops a column.
        if until_steady_state and check_lateral_inhibition_steady:
            li_cols = [
                c for c in ("notch_level", "delta_level", "repressor_level")
                if c in self.eptm.face_df.columns
            ]
        else:
            li_cols = []

        current_t = float(self.prev_t)
        next_save_t = current_t + save_interval

        # Record the initial state.
        self._record_at(current_t, dt)

        pbar = tqdm(total=tf - current_t, unit="t", smoothing=0.05)
        try:
            while current_t < tf:
                pos = self.current_pos
                # Snapshot lateral-inhibition levels at the START of
                # the iteration so the post-manager comparison sees
                # the FULL change (mechanics + differentiation step).
                # Only captured when we'll actually use it — keeps the
                # default code path zero-overhead.
                if li_cols:
                    old_li = self.eptm.face_df[li_cols].to_numpy().copy()
                else:
                    old_li = None
                try:
                    new_pos = solve_ivp(
                        self.ode_func, (0, dt), pos, t_eval=[dt]
                    ).y[:, 0]
                except Exception as exc:  # numerical failure inside scipy
                    log.warning(
                        "solve_ivp failed at t=%g (dt=%g): %s; shrinking dt",
                        current_t, dt, exc,
                    )
                    dt *= 0.5
                    if dt < dt_min:
                        raise RuntimeError(
                            f"dt fell below {dt_min:.3e} (initial {initial_dt:.3e}) "
                            f"after solve_ivp failure at t={current_t:g}: {exc}"
                        )
                    continue

                # --- B: per-step displacement check ---
                # Compute per-vertex displacement (max abs component) so
                # we can identify WHICH vertex is moving too fast.
                delta = (new_pos - pos).reshape(-1, self.eptm.dim)
                per_vert_disp = np.abs(delta).max(axis=1)
                worst_idx = int(per_vert_disp.argmax())
                disp = float(per_vert_disp[worst_idx])
                # active_verts holds positional indices into vert_df; the
                # vertex LABEL is what the user actually wants to look up.
                if worst_idx < len(self.eptm.active_verts):
                    worst_vert_label = int(self.eptm.active_verts[worst_idx])
                else:
                    worst_vert_label = worst_idx
                if disp > max_displacement:
                    dt *= 0.5
                    if dt < dt_min:
                        # Also report the worst-offending vertex when we
                        # finally give up so the user can inspect it.
                        try:
                            wx = float(self.eptm.vert_df.iloc[worst_idx]["x"])
                            wy = float(self.eptm.vert_df.iloc[worst_idx]["y"])
                            pos_msg = f" at ({wx:.4f}, {wy:.4f})"
                        except Exception:
                            pos_msg = ""
                        self._record_at(current_t, dt)
                        raise RuntimeError(
                            f"dt fell below {dt_min:.3e} (initial {initial_dt:.3e}) "
                            f"at t={current_t:g}: vertex {worst_vert_label}{pos_msg} "
                            f"moved {disp:.3e} (cap {max_displacement:.3e})"
                        )
                    # Upgraded from DEBUG to WARNING so the user sees
                    # which vertex is the troublemaker without enabling
                    # debug logging.
                    log.warning(
                        "step rejected at t=%g: vertex %d moved %.3e (cap %.3e); dt -> %.3e",
                        current_t, worst_vert_label, disp, max_displacement, dt,
                    )
                    continue

                # Optional legacy clipping (kept for back-compat with
                # callers who set self.bounds explicitly).
                if self.bounds is not None:
                    new_pos = self.clip_new_pos(pos, new_pos)
                self.set_pos(new_pos)

                # --- C: negative-area safety net ---
                # If an edge crossed through a face, that face's signed
                # area flips sign. Revert and retry with smaller dt.
                if (self.eptm.face_df["area"] < 0).any():
                    dt *= 0.5
                    if dt < dt_min:
                        self._record_at(current_t, dt)
                        raise RuntimeError(
                            f"dt fell below {dt_min:.3e} (initial {initial_dt:.3e}) "
                            f"at t={current_t:g}: edge crossing produced negative area"
                        )
                    log.warning(
                        "step rejected at t=%g: negative face area detected; "
                        "dt -> %.3e", current_t, dt,
                    )
                    self.set_pos(pos)  # restore previous positions
                    continue

                # Step accepted.
                current_t += dt
                self.eptm.settings["dt"] = dt
                self.prev_t = current_t

                # Topology events / differentiation / random forces.
                if self.manager is not None:
                    if quasi_static:
                        if (np.abs(pos - new_pos) < quasi_static_threshold).all():
                            self.manager.append(
                                self.inner.lateral_inhibition_model
                                .get_length_dependent_differentiation_function(
                                    dt=current_t - last_differentiation_t,
                                    quasi_static=True,
                                )
                            )
                            last_differentiation_t = current_t
                    self.eptm.update_after_each_time_step()
                    self.manager.execute(self.eptm)
                    self.geom.update_all(self.eptm)
                    self._refresh_active_verts()
                    self.manager.update()

                # Capture the topology-change flag BEFORE the handler
                # below resets it — the steady-state check (further
                # down) needs to know whether this step changed the
                # topology, since divisions/delaminations break the
                # element-wise old/new LI comparison.
                topo_changed_this_step = bool(self.eptm.topo_changed)
                if self.eptm.topo_changed:
                    log.info("Topology changed")
                    if on_topo_change is not None:
                        on_topo_change(*topo_change_args)
                    self.eptm.topo_changed = False

                # --- Steady-state stop ---
                # Evaluated only when the caller opted in. Both
                # criteria default to "satisfied" when their flag is
                # disabled, so the AND combination falls back to the
                # single enabled check (or to "stop immediately" if
                # the caller enabled neither — which is silly but
                # explicitly the user's call).
                if until_steady_state:
                    if check_mechanical_steady:
                        mech_ok = bool(
                            np.abs(new_pos - pos).max() < quasi_static_threshold
                        )
                    else:
                        mech_ok = True

                    if check_lateral_inhibition_steady and li_cols:
                        new_li = self.eptm.face_df[li_cols].to_numpy()
                        if topo_changed_this_step or new_li.shape != old_li.shape:
                            # Topology changed → face row count
                            # differs, element-wise compare is
                            # meaningless. Wait for things to settle.
                            li_ok = False
                        else:
                            li_ok = bool(
                                np.abs(new_li - old_li).max()
                                < lateral_inhibition_threshold
                            )
                    elif check_lateral_inhibition_steady:
                        # Caller asked for LI check but the sheet
                        # carries none of the LI columns — that's a
                        # configuration error, so don't pretend things
                        # are steady on no data. Treat as never-steady.
                        li_ok = False
                    else:
                        li_ok = True

                    if mech_ok and li_ok:
                        log.info(
                            "steady state reached at t=%g "
                            "(mech_ok=%s, li_ok=%s)",
                            current_t, mech_ok, li_ok,
                        )
                        # Record this state before breaking so the
                        # history's final frame is the steady-state
                        # snapshot.
                        self._record_at(current_t, dt)
                        break

                # --- Constant-time history saving ---
                # Save when we cross one or more save_interval boundaries,
                # using the actual current_t as the timestamp. If a large
                # dt skipped multiple boundaries, save once at current_t
                # (the state hasn't changed across those skipped points).
                if current_t >= next_save_t:
                    self._record_at(current_t, dt)
                    # Advance past the current point so we don't double-save.
                    while next_save_t <= current_t:
                        next_save_t += save_interval

                # Ratchet dt back toward the initial value as things calm down.
                dt = min(dt * dt_increase_factor, initial_dt)

                # tqdm progress in SIMULATION time, not iterations.
                pbar.update(current_t - pbar.n)
        finally:
            pbar.close()

        # Make sure the final state is always recorded, even if it
        # doesn't land on a save boundary.
        self._record_at(current_t, dt)


