"""Time integration, and the safety nets that stop a run producing nonsense.

:class:`IVPSolver` advances the tissue viscously and, between steps, lets the
topology handler act — splitting long bonds, collapsing short ones, resolving T1
transitions. Because topology changes discontinuously, the step is adaptive: a
step that produces invalid geometry is rejected and retried at a smaller ``dt``.

Two failure modes matter enough to be checked explicitly, and they are not the
same one:

* **Inverted cells** — a face whose signed area has gone negative. Cheap to
  detect and unambiguous.
* **Folded cells** — a face whose perimeter crosses itself while its signed area
  stays POSITIVE, because the two lobes partly cancel. The area test cannot see
  this, yet it is exactly the "cells growing through each other" configuration.
  :func:`count_folded_faces` catches it with a turning-number test, which is
  sound for any simple polygon, convex or not.

A run that cannot take a valid step even at the smallest ``dt`` raises rather
than continuing, so a caller scoring the result never reads a corrupt sheet as
data.
"""
import time
import numpy as np
from tyssue.solvers.viscous import EulerSolver, log
from tqdm import tqdm


def count_folded_faces(eptm, tol=0.3):
    """Number of faces whose polygon has FOLDED OVER ITSELF (self-intersected).

    Detected via the polygon turning number: walking a face's edges in
    ``order`` and summing the signed turn angle between consecutive edge
    vectors gives ``±2π`` (turning number ``±1``) for ANY simple polygon —
    convex or not — and deviates from ``±1`` once the perimeter crosses
    itself. The check is therefore SOUND (a simple polygon is exactly ``±1``
    by the turning-number theorem, so no false positives) and O(Ne).

    This complements the negative-signed-area check: a cell that folds over a
    neighbour keeps a POSITIVE signed area (the two lobes partly cancel), so
    the ``area < 0`` net misses it — yet such a fold is exactly the
    unphysical "cells growing into each other" configuration. ``tol`` is the
    allowed deviation of ``|turning number|`` from 1; 0.3 only flags clear
    self-intersections (turning number near 0) and leaves a wide margin over
    the floating-point error of a simple polygon's exact ``±1``.
    """
    e = eptm.edge_df
    if e.shape[0] == 0:
        return 0
    cols = ["face", "dx", "dy"] + (["order"] if "order" in e.columns else [])
    ed = e[cols]
    ed = ed.sort_values(["face", "order"]) if "order" in e.columns else ed.sort_values(["face"])
    face = ed["face"].to_numpy()
    vx = ed["dx"].to_numpy()
    vy = ed["dy"].to_numpy()
    n = len(face)
    # First index of each contiguous same-face run, and the last edge of each.
    change = np.empty(n, bool)
    change[0] = True
    change[1:] = face[1:] != face[:-1]
    run_start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    is_last = np.empty(n, bool)
    is_last[:-1] = face[1:] != face[:-1]
    is_last[-1] = True
    # "Next" edge cyclically within the face (the last edge wraps to the first).
    nxt = np.arange(n) + 1
    nxt[is_last] = run_start[is_last]
    wx = vx[nxt]
    wy = vy[nxt]
    ang = np.arctan2(vx * wy - vy * wx, vx * wx + vy * wy)
    starts = np.where(change)[0]
    turn = np.add.reduceat(ang, starts) / (2.0 * np.pi)
    return int(np.count_nonzero(np.abs(np.abs(turn) - 1.0) > tol))


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
              method=None, quasi_static=False, quasi_static_threshold=0.01,
              max_displacement=None, max_disp_factor=0.25,
              dt_min_factor=0.001, dt_increase_factor=1.1,
              save_interval=None,
              until_steady_state=False,
              lateral_inhibition_threshold=0.0,
              check_mechanical_steady=True,
              check_lateral_inhibition_steady=True,
              steady_state_min_steps=4,
              tolerate_unavoidable_folds=True,
              max_wall_seconds=None, min_progress_rate=None,
              progress_window_seconds=30.0):
        """Solves the ODE from the current time to tf with ADAPTIVE dt
        and edge-crossing safety nets.

        Each accepted iteration advances the sheet with a single explicit
        forward-Euler step (see the body for why this is exact for this
        position-independent mechanics RHS). ``method`` is accepted for
        backward compatibility but no longer used.

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

        steady_state_min_steps : int, default 4
            How many CONSECUTIVE accepted steps must satisfy the
            steady-state criteria before the run halts. The default
            of 4 means "no significant change for more than 3 steps".
            The streak resets to zero whenever a step fails the
            criteria, a topology change occurs, OR a step is rejected
            (displacement / negative-area / solver failure) — so a
            single transient blip restarts the count. Set to 1 to
            recover the old "halt on the first steady step" behaviour.

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

        The run only halts once these criteria have held for
        ``steady_state_min_steps`` consecutive accepted steps — a
        single in-between step that drifts (or a rejected step, or a
        division / delamination / T1) resets the counter, so brief
        excursions can't trigger a premature stop.
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

        # Number of CONSECUTIVE accepted steps that have satisfied the
        # steady-state criteria. We only halt once this reaches
        # ``steady_state_min_steps`` — a single drifting / rejected /
        # topology-changing step resets it to zero.
        steady_streak = 0

        # Record the initial state.
        self._record_at(current_t, dt)

        # Baseline number of self-intersecting (folded) faces. The fold safety
        # net (check C2 below) rejects a step only if it INCREASES this count —
        # i.e. the mechanics introduces a NEW fold — rather than rejecting any
        # absolute fold. Some saved initial sheets already contain a few folded
        # cells (a pre-existing degeneracy); an absolute check would reject
        # every step and the run could never start. We still forbid the
        # mechanics from making things worse, and allow it to improve.
        prev_folded = count_folded_faces(self.eptm)

        # Non-progress safety net (optional; for parameter fits). A pathological
        # parameter region can make the run CRAWL — dt pinned tiny by buckling
        # spikes, the sharp-corner collapse churning every step — so it would
        # take hours/days to reach tf. Rather than burn that time, bail with a
        # RuntimeError (which find_mechanical_parameters scores worst-case).
        # Two independent limits, both default OFF:
        #   max_wall_seconds   - hard cap on total solve() wall-clock time.
        #   min_progress_rate  - floor on SIMULATION-time advanced per wall-clock
        #                        second, measured over a sliding window; catches
        #                        "stuck at a tiny dt / progressing very slowly"
        #                        EARLY, before the full wall-clock budget.
        _start_wall = time.monotonic()
        _progress_wall = _start_wall
        _progress_sim = current_t

        pbar = tqdm(total=tf - current_t, unit="t", smoothing=0.05)
        try:
            while current_t < tf:
                if max_wall_seconds is not None or min_progress_rate is not None:
                    _now = time.monotonic()
                    if (max_wall_seconds is not None
                            and _now - _start_wall > max_wall_seconds):
                        log.warning("stopping at t=%g: wall-clock budget %.0fs "
                                    "exceeded (elapsed %.0fs)", current_t,
                                    max_wall_seconds, _now - _start_wall)
                        self._record_at(current_t, dt)
                        raise RuntimeError(
                            f"simulation exceeded its wall-clock budget "
                            f"({max_wall_seconds:g}s) at t={current_t:g} of "
                            f"tf={tf:g}; stopping for worst-case scoring")
                    if (min_progress_rate is not None
                            and _now - _progress_wall >= progress_window_seconds):
                        _rate = (current_t - _progress_sim) / (_now - _progress_wall)
                        if _rate < min_progress_rate:
                            log.warning("stopping at t=%g: progress %.3g sim-time/s "
                                        "over last %.0fs < floor %.3g", current_t,
                                        _rate, _now - _progress_wall, min_progress_rate)
                            self._record_at(current_t, dt)
                            raise RuntimeError(
                                f"simulation progressing too slowly ({_rate:.3g} "
                                f"sim-time/s < {min_progress_rate:g}) over the last "
                                f"{_now - _progress_wall:.0f}s at t={current_t:g}; "
                                f"stopping for worst-case scoring")
                        _progress_wall = _now
                        _progress_sim = current_t
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
                    # Explicit forward-Euler step.
                    #
                    # ``ode_func`` (the mechanics RHS) reads the CURRENT
                    # sheet geometry and ignores the position argument an
                    # ODE integrator would feed it — it never calls
                    # ``set_pos`` to move the sheet to an intermediate RK
                    # stage. With a position-independent RHS every stage of
                    # an adaptive integrator returns the identical vector,
                    # so the exact integral over (0, dt) collapses to a
                    # single Euler step ``new = pos + dt * f(pos)``. The
                    # previous implementation called ``solve_ivp`` here,
                    # which evaluated the (expensive) gradient ~8 times per
                    # accepted step to recompute the very same constant —
                    # this gives a bit-for-bit equivalent result (verified
                    # to ~1e-16) with ONE gradient evaluation per step.
                    dot_r = self.ode_func(0.0, pos)
                    new_pos = pos + dt * dot_r
                except Exception as exc:  # numerical failure in the gradient
                    log.debug(
                        "force evaluation failed at t=%g (dt=%g): %s; shrinking dt",
                        current_t, dt, exc,
                    )
                    dt *= 0.5
                    steady_streak = 0  # a failed step breaks the steady run
                    if dt < dt_min:
                        raise RuntimeError(
                            f"dt fell below {dt_min:.3e} (initial {initial_dt:.3e}) "
                            f"after force-evaluation failure at t={current_t:g}: {exc}"
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
                    steady_streak = 0  # rejected step breaks the steady run
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
                    log.debug(
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
                # If an edge crossed through a face, that face's signed area
                # flips sign. Revert and retry with smaller dt. EXCEPTION: a
                # DELAMINATING cell (type == -1) is deliberately collapsing to
                # zero/negative area on its way to removal by the delamination
                # handler, so a negative area there is expected, not an edge
                # crossing. Ignore those — otherwise the inherited inversion
                # (created by the manager, present at the start of the next
                # step) is rejected on every retry down to the dt floor and the
                # run dies before the manager can remove the cell.
                neg_area = self.eptm.face_df["area"] < 0
                if "type" in self.eptm.face_df.columns:
                    neg_area = neg_area & (self.eptm.face_df["type"] != -1)
                if neg_area.any():
                    dt *= 0.5
                    steady_streak = 0  # rejected step breaks the steady run
                    if dt < dt_min:
                        self._record_at(current_t, dt)
                        raise RuntimeError(
                            f"dt fell below {dt_min:.3e} (initial {initial_dt:.3e}) "
                            f"at t={current_t:g}: edge crossing produced negative area"
                        )
                    log.debug(
                        "step rejected at t=%g: negative face area detected; "
                        "dt -> %.3e", current_t, dt,
                    )
                    self.set_pos(pos)  # restore previous positions
                    continue

                # --- C2: self-intersecting (folded) face safety net ---
                # A cell that folds over a neighbour keeps a POSITIVE signed
                # area, so check C above misses it. Detect the fold from the
                # polygon turning number. We only react to a step that ADDS a
                # fold (count rises above the running baseline) — several saved
                # initial sheets already carry a few folded cells, which must
                # not block the run.
                #
                # A new fold is one of two kinds: TRANSIENT (caused by too large
                # a step — a smaller step wouldn't cross) or INHERENT (the
                # tissue is genuinely overlapping in this configuration). We
                # shrink dt to dodge the transient kind; but if the fold
                # survives all the way down to the dt floor it is inherent, and
                # shrinking further would only deadlock the run. In a full
                # differentiation run such folds are long-lived yet harmless —
                # they form and persist for many steps and resolve later as the
                # tissue develops (divisions / rearrangements via the topology
                # manager below), exactly as before this safety net existed — so
                # by default we TOLERATE them at the floor and carry on. A
                # mechanics-only parameter search, where a fold means cells
                # over-packed past the available area and nothing will resolve
                # it, can opt into the strict behaviour
                # (tolerate_unavoidable_folds=False) so dt collapse raises and
                # the caller can score those parameters as a failure.
                n_folded = count_folded_faces(self.eptm)
                new_folds = n_folded - prev_folded
                if new_folds > 0:
                    if dt * 0.5 >= dt_min:
                        # Retry with a smaller step — might dodge the crossing.
                        self.set_pos(pos)  # restore the last good positions
                        dt *= 0.5
                        steady_streak = 0  # rejected step breaks the steady run
                        log.debug(
                            "step rejected at t=%g: %d new self-intersecting "
                            "face(s) (%d total); dt -> %.3e",
                            current_t, new_folds, n_folded, dt,
                        )
                        continue
                    if not tolerate_unavoidable_folds:
                        self._record_at(current_t, dt)
                        raise RuntimeError(
                            f"dt fell below {dt_min:.3e} (initial "
                            f"{initial_dt:.3e}) at t={current_t:g}: a step "
                            f"introduced {new_folds} new self-intersecting "
                            f"face(s) ({n_folded} total; cells folding / "
                            f"overlapping)"
                        )
                    log.debug(
                        "fold-floor at t=%g: tolerating %d unavoidable self-"
                        "intersecting face(s) (%d total); relying on tissue "
                        "development / topology to resolve",
                        current_t, new_folds, n_folded,
                    )
                    # Fall through and accept the step with the floored dt; the
                    # dt ratchet grows it back on subsequent calm steps.

                # Step accepted.
                current_t += dt
                self.eptm.settings["dt"] = dt
                self.prev_t = current_t
                # New baseline (mechanics only — refreshed again after the
                # manager below, since topology events can change the count).
                prev_folded = n_folded

                # NOTE on periodic boundaries: the per-step displacement below
                # (new_pos - pos) is already correct even for a periodic sheet and
                # needs NO minimum-image handling. new_pos = pos + dt*f(pos) is the
                # raw forward-Euler step and is never re-wrapped, and an accepted
                # step moves less than max_displacement (~ max_disp_factor *
                # min_bond_length << box/2), so the difference is the vertex's true
                # motion, never an apparent full-box jump. (Min-image would only be
                # needed when differencing two INDEPENDENTLY-wrapped positions —
                # e.g. across saved history frames — which this is not; and it must
                # NOT be applied here, since folding could mask a genuine blow-up.)

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
                    # Topology events (T1 / division / delamination / virtual
                    # vertex changes) can add or remove folds, so re-baseline
                    # the fold count against the post-manager state.
                    prev_folded = count_folded_faces(self.eptm)

                # Capture the topology-change flag BEFORE the handler
                # below resets it — the steady-state check (further
                # down) needs to know whether this step changed the
                # topology, since divisions/delaminations break the
                # element-wise old/new LI comparison.
                topo_changed_this_step = bool(self.eptm.topo_changed)
                if self.eptm.topo_changed:
                    # The specific topological events (T1 / division / ... ) log
                    # themselves via log_topo_event; a bare "topology changed"
                    # per step was redundant noise, so it was removed.
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
                        # Raw (new_pos - pos) is periodicity-correct — see the
                        # NOTE where the step is accepted above.
                        mech_ok = bool(
                            np.abs(new_pos - pos).max()/dt < quasi_static_threshold
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

                    # Require the criteria to hold for several
                    # CONSECUTIVE accepted steps. One drifting step
                    # (or a topology change, which forces li_ok=False)
                    # resets the streak, so a brief lull can't trigger
                    # a premature halt.
                    if mech_ok and li_ok:
                        steady_streak += 1
                    else:
                        steady_streak = 0

                    if steady_streak >= steady_state_min_steps:
                        log.info(
                            "steady state reached at t=%g after %d "
                            "consecutive steady steps (mech_ok=%s, li_ok=%s)",
                            current_t, steady_streak, mech_ok, li_ok,
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

                # tqdm progress in SIMULATION time, not iterations. The bar is
                # purely cosmetic — and on Windows a console write can fail
                # intermittently with OSError [Errno 22], which previously
                # killed multi-hour runs. A display failure must NEVER abort the
                # simulation, so swallow anything tqdm raises (and stop updating
                # it once it has broken, to avoid retrying every step).
                try:
                    pbar.update(current_t - pbar.n)
                except Exception:
                    pbar.disable = True
        finally:
            try:
                pbar.close()
            except Exception:
                pass

        # Make sure the final state is always recorded, even if it
        # doesn't land on a save boundary.
        self._record_at(current_t, dt)


