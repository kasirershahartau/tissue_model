"""Self-contained Gaussian-process Bayesian optimization (numpy + scipy only).

Intended for *expensive, noisy, low-dimensional, box-constrained* black-box
objectives — exactly the regime of fitting a handful of simulation parameters
to data, where each evaluation is a full simulation and finite-difference
gradient descent is both slow and unreliable.

The optimizer

* seeds a space-filling Latin-Hypercube initial design,
* fits a Gaussian process surrogate (ARD Matern-5/2 kernel + a learned noise
  term, so stochastic objectives are handled),
* and picks each next point by maximizing Expected Improvement.

Only :func:`minimize` is needed by callers; everything else is internal.
"""

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, LinAlgError
from scipy.optimize import minimize as _scipy_minimize
from scipy.stats import norm, qmc


_SQRT5 = np.sqrt(5.0)


def _matern52(sq_dist):
    """Matern-5/2 correlation from a squared (already length-scaled) distance."""
    r = np.sqrt(np.maximum(sq_dist, 0.0))
    return (1.0 + _SQRT5 * r + (5.0 / 3.0) * sq_dist) * np.exp(-_SQRT5 * r)


class _GaussianProcess:
    """Minimal GP regressor: ARD Matern-5/2 kernel, Gaussian noise, MLE-fit
    hyper-parameters. Works on inputs already scaled to the unit cube and
    internally standardizes the targets for numerical stability."""

    def __init__(self, X, y):
        self.X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if y.std() > 1e-12 else 1.0
        self.y = (y - self.y_mean) / self.y_std
        self.n, self.d = self.X.shape

    def _kernel(self, A, B, lengthscales, output_scale):
        A_s = A / lengthscales
        B_s = B / lengthscales
        sq = (np.sum(A_s ** 2, axis=1)[:, None]
              + np.sum(B_s ** 2, axis=1)[None, :]
              - 2.0 * A_s @ B_s.T)
        return output_scale ** 2 * _matern52(sq)

    def _neg_log_marginal_likelihood(self, theta):
        output_scale = np.exp(theta[0])
        lengthscales = np.exp(theta[1:1 + self.d])
        noise = np.exp(theta[-1])
        K = self._kernel(self.X, self.X, lengthscales, output_scale)
        K[np.diag_indices_from(K)] += noise ** 2 + 1e-8
        try:
            L = cholesky(K, lower=True)
        except (LinAlgError, np.linalg.LinAlgError):
            return 1e25
        alpha = cho_solve((L, True), self.y)
        return float(0.5 * self.y @ alpha
                     + np.sum(np.log(np.diag(L)))
                     + 0.5 * self.n * np.log(2.0 * np.pi))

    def fit(self, n_restarts=8, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        # Search log-hyper-parameters over sensible ranges (inputs live in the
        # unit cube, targets are standardized).
        lower = np.array([np.log(1e-2)] + [np.log(1e-2)] * self.d + [np.log(1e-3)])
        upper = np.array([np.log(1e1)] + [np.log(1e1)] * self.d + [np.log(1.0)])
        bounds = list(zip(lower, upper))
        best = None
        for _ in range(n_restarts):
            x0 = rng.uniform(lower, upper)
            res = _scipy_minimize(self._neg_log_marginal_likelihood, x0,
                                  method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        self._output_scale = np.exp(self.theta[0])
        self._lengthscales = np.exp(self.theta[1:1 + self.d])
        self._noise = np.exp(self.theta[-1])
        K = self._kernel(self.X, self.X, self._lengthscales, self._output_scale)
        K[np.diag_indices_from(K)] += self._noise ** 2 + 1e-8
        self._L = cholesky(K, lower=True)
        self._alpha = cho_solve((self._L, True), self.y)
        return self

    def predict(self, Xstar, standardized=False):
        """Posterior mean and std at ``Xstar``. ``standardized=True`` returns
        them in the internal zero-mean/unit-var target space (used by the
        acquisition); otherwise they are mapped back to the original scale."""
        Xstar = np.atleast_2d(np.asarray(Xstar, float))
        Ks = self._kernel(self.X, Xstar, self._lengthscales, self._output_scale)
        mu = Ks.T @ self._alpha
        v = solve_triangular(self._L, Ks, lower=True)
        var = self._output_scale ** 2 - np.sum(v ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 1e-12))
        if standardized:
            return mu, std
        return mu * self.y_std + self.y_mean, std * self.y_std


def _expected_improvement(Xstar, gp, best_f_standardized, xi=0.01):
    """EI for *minimization*, computed in the GP's standardized target space."""
    mu, std = gp.predict(Xstar, standardized=True)
    improvement = best_f_standardized - mu - xi
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    ei[std < 1e-12] = 0.0
    return ei


def _propose_location(gp, best_f_standardized, d, rng,
                      n_samples=2000, n_restarts=10, xi=0.01):
    """Maximize EI over the unit cube: dense random scan + L-BFGS-B polish."""
    candidates = rng.uniform(0.0, 1.0, size=(n_samples, d))
    ei = _expected_improvement(candidates, gp, best_f_standardized, xi)
    order = np.argsort(-ei)
    best_x = candidates[order[0]]
    best_ei = ei[order[0]]

    def neg_ei(x):
        return -float(_expected_improvement(x[None, :], gp, best_f_standardized, xi)[0])

    for idx in order[:n_restarts]:
        res = _scipy_minimize(neg_ei, candidates[idx], method="L-BFGS-B",
                              bounds=[(0.0, 1.0)] * d)
        if -res.fun > best_ei:
            best_ei = -res.fun
            best_x = res.x
    return np.clip(best_x, 0.0, 1.0)


def minimize(objective, bounds, n_calls=40, n_initial_points=10,
             random_state=0, xi=0.01, verbose=True, x0=None,
             return_surrogate=False):
    """Bayesian-optimization minimizer.

    Parameters
    ----------
    objective : callable(np.ndarray (d,)) -> float
        The (possibly noisy) function to MINIMIZE. Called ``n_calls`` times.
    bounds : sequence of (low, high)
        Box constraints, one pair per dimension.
    n_calls : int
        Total objective evaluations (initial design + BO iterations).
    n_initial_points : int
        Size of the Latin-Hypercube initial design (must be >= 2).
    random_state : int
        Seed for the design, GP restarts and acquisition scan.
    xi : float
        Exploration margin for Expected Improvement (in standardized units).
    x0 : array-like, optional
        An extra point (e.g. a hand-tuned guess) evaluated first, on top of
        the initial design.

    return_surrogate : bool, default False
        When True, fit a FINAL Gaussian process on ALL evaluated points and add
        two extra keys to the result: ``gp`` (the fitted :class:`_GaussianProcess`,
        living on the unit cube with standardized targets) and ``surrogate`` — a
        convenience callable ``points_in_original_units -> (mean, std)`` giving
        the estimated objective landscape (posterior mean and uncertainty)
        anywhere in the box. Both are ``None`` if the final GP fit fails.

    Returns
    -------
    dict with keys ``x`` (best params), ``fun`` (best value), ``X`` (all
    evaluated points, shape (n_calls, d)) and ``y`` (all values); plus ``gp`` /
    ``surrogate`` when ``return_surrogate`` is set.
    """
    bounds = np.asarray(bounds, float)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    rng = np.random.default_rng(random_state)

    def to_unit(X):
        return (np.atleast_2d(X) - lo) / span

    def from_unit(U):
        return lo + np.atleast_2d(U) * span

    n_initial_points = max(2, min(n_initial_points, n_calls))
    design = qmc.LatinHypercube(d=d, seed=random_state).random(n_initial_points)
    init_points = from_unit(design)
    if x0 is not None:
        init_points = np.vstack([np.atleast_2d(np.asarray(x0, float)), init_points])

    X_unit, y = [], []
    for i, point in enumerate(init_points):
        if len(y) >= n_calls:
            break
        value = float(objective(point))
        X_unit.append(to_unit(point)[0])
        y.append(value)
        if verbose:
            print("[BO init %d/%d] x=%s  f=%.6g" % (i + 1, len(init_points), np.array2string(point, precision=4), value))

    while len(y) < n_calls:
        gp = _GaussianProcess(np.array(X_unit), np.array(y)).fit(rng=rng)
        best_f_std = (min(y) - gp.y_mean) / gp.y_std
        u_next = _propose_location(gp, best_f_std, d, rng, xi=xi)
        x_next = from_unit(u_next)[0]
        value = float(objective(x_next))
        X_unit.append(u_next)
        y.append(value)
        if verbose:
            print("[BO iter %d/%d] x=%s  f=%.6g  best=%.6g"
                  % (len(y) - len(init_points), n_calls - len(init_points),
                     np.array2string(x_next, precision=4), value, min(y)))

    y = np.array(y)
    X = from_unit(np.array(X_unit))
    best_idx = int(np.argmin(y))
    result = {"x": X[best_idx], "fun": float(y[best_idx]), "X": X, "y": y}

    if return_surrogate:
        # Fit a FINAL GP on ALL evaluations so callers can query the estimated
        # objective landscape (posterior mean + std) anywhere in the box. The GP
        # lives on the unit cube with standardized targets; ``surrogate`` wraps
        # the input scaling so callers pass ORIGINAL-unit parameter points and
        # get objective-unit predictions back.
        gp_final, surrogate = None, None
        try:
            gp_final = _GaussianProcess(np.array(X_unit), y).fit(rng=rng)

            def surrogate(points, _gp=gp_final):
                return _gp.predict(to_unit(np.asarray(points, float)),
                                   standardized=False)
        except Exception:
            gp_final, surrogate = None, None
        result["gp"] = gp_final
        result["surrogate"] = surrogate

    return result
