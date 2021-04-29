"""Stochastic game base class

dsGame: base class for solving stochastic game


dsGameSolver:
Python software for computing stationary equilibria
of finite discounted stochastic games.

Copyright (C) 2018  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it
and/or modify it under the terms of the MIT License.
"""


import sys
from time import perf_counter
from datetime import timedelta

import numpy as np


# %% base class for solving stochastic game


class dsGame:
    """Base class for storing and solving a stochastic game.

    Given:  specification of stochastic game:
        1) payoffs[state, player, action_profile]
        2) transition_probs[from_state, player, action_profile, to_state]
        3) discount_factors[player]

    Wanted:  stationary equilibrium:
        1) strategies[state, player, action] -> action probabilities
        2) state_values[state, player] -> expected discounted payoffs

    Idea:  Construct homotopy and trace path to equilibrium, using HomCont.

    Dimension of game:
        num_s: number of states
        num_p: number of players
        nums_a[s,p]: number of actions for player p in state s

    Main inputs
    -----------
    payoff_matrices : list of np.ndarray
        One payoff matrix for each state s=1,...,S:
            [payoff_matrix_1, ..., payoff_matrix_S].
        The payoff for player p in state s under action profile (a1,...,aP) is given by
            payoff_matrix_state_s[p,a1,a2,...,aP].
    transition_matrices : list of np.ndarray
        One transition matrix for each state s=1,...,S:
            [transition_matrix_state_1, ..., transition_matrix_state_S].
        The probability for player p to transition from state s to state s'
            under action profile (a1,...,aP) is given by
            transition_matrix_state_s


    J : callable, optional
        Jacobian matrix.
        If not provided by user, a finite difference approximation is used.
        (Requires package numdifftools.)

    t_target : float, optional
        Target value of the homotopy parameter, by default np.inf.
        If np.inf, iteration continues until all components of x converge.
    sign: int, +1, -1, or None
        Orientation in which the path is traced.
        If None (default), orientation is chosen s.t. tracing starts towards t_target.
        TODO: comment?
    max_steps : float, optional
        Maximum number of predictor-corrector iterations, by default np.inf.
    x_transformer : callable, optional
        Transformation of x to check for convergence, by default lambda x : x.
        TODO: comment?
    verbose : bool, optional
        Whether to show progress reports, by default False.
    store_path: bool, optional
        If True, the path traversed is stored as HomPath, accessible as this.path.
        Allows to graph the path, and also to restart continuation at previous points.
        Slows down solution a bit, and increases memory footprint.


    parameters : dict
        Collection of parameters for path tracing, which will override the defaults.
        May alternatively be passed as kwargs.
        Method .set_parameters() allows to adjust parameters after construction,
        again passed as dict or kwargs.

        The parameters are:
        ----------
        x_tol : float
            Continuation is considered converged once x has stabilized, i.e.
            max(|x_new-x_old|) / ds < x_tol.
            x is transformed by x_transformer for this criterion, if provided.
            Active only if t_target = np.inf. Defaults to 1e-7.
        t_tol : float
            Continuation is considered converged once |t - t_target| < t_tol.
            Active only if t_target < np.inf. Defaults to 1e-7.
        ds0 : float
            Initial step size, defaults to 0.01.
        ds_infl : float
            Step size inflation factor, defaults to 1.2.
        ds_defl : float
            Step size deflation factor, defaults to 0.5.
        ds_min : float
            Minimum step size, defaults to 1e-9.
        ds_max : float
            Maximum step size, defaults to 1000.
        H_tol : float
            Convergence criterion used for corrector step: max(H(y_corr)) < H_tol.
            Defaults to 1e-7.
        corr_steps_max : int
            Maximum number of corrector steps, defaults to 20.
        corr_dist_max : float
            Maximum distance of corrector steps, defaults to 0.3.
        corr_ratio_max : float
            Maximum ratio between consecutive corrector steps, defaults to 0.3.
        detJ_change_max : float
            Maximum relative change of determinant of augmented Jacobian between
            consecutive predictor-corrector steps.
            Predictor-corrector step is discarded unless
            detJ_change_max < |detJ_new|/|det_J_old| < 1/detJ_change_max.
            (Large changes in augmented determinant indicate potential segment jumping.)
            Defaults to 0.5.
        bifurc_angle_min : float
            Minimum angle (in degrees) between two consecutive predictor tangents
            to be considered a bifurcation, defaults to 177.5.
            If a bifurcations is crossed, path orientation is swapped.
            TODO: alternatively: change in sign of determinant of augmented Jacobian?
        transvers_angle_max : float
            Minimum angle (in degrees) between tangent at starting point and t-axis
            to raise a warning that transversality may be violated. Defaults to 87.5.
    """

    def __init__(
        self,
        H: callable,
        y0: np.ndarray,
        J: callable = None,
        t_target: float = np.inf,
        max_steps: float = np.inf,
        sign: int = None,
        x_transformer: callable = lambda x: x,
        verbose: bool = False,
        store_path: bool = False,
        parameters: dict = {},
        **kwargs,
    ):

        self.H_func = H
        if J is not None:
            self.J_func = J
        else:
            try:
                import numdifftools as nd
            except ModuleNotFoundError:
                print("If J is not provided by user, package numdifftools is required.")
                raise
            else:
                self.J_func = nd.Jacobian(H)
        self.y = y0

        self.t_target = t_target
        self.max_steps = max_steps
        self.x_transformer = x_transformer
        self.verbose = verbose

        # set default parameters
        self.x_tol = 1e-7
        self.t_tol = 1e-7
        self.ds0 = 0.01
        self.ds_infl = 1.2
        self.ds_defl = 0.5
        self.ds_min = 1e-9
        self.ds_max = 1000
        self.H_tol = 1e-7
        self.corr_steps_max = 20
        self.corr_dist_max = 0.3
        self.corr_ratio_max = 0.3
        self.detJ_change_max = 0.5
        self.bifurc_angle_min = 177.5
        self.transvers_angle_max = 87.5

        # overwrite defaults with user-provided parameters if present
        self.set_parameters(parameters, **kwargs)

        if sign is not None and sign != 0:
            self.sign = np.sign(sign)
        else:
            self.set_greedy_sign()

        self.tangent_old = self.tangent

        self.store_path = store_path
        if store_path:
            self.path = HomPath(dim=len(self.y), x_transformer=self.x_transformer)
            self.path.update(y=self.y, s=0, cond=self.cond, sign=self.sign, step=0)

        # attributes to be used later
        self.ds = self.ds0
        self.iteration = 0
        self.s = 0
        self.corrector_success = False
        self.converged = False
        self.failed = False

        self.H_pred = None
        self.y_corr = None
        self.J_corr = None
        self.corr_step = None

        # storage vars and flags for cached attributes
        self._y_pred = None
        self._J = None
        self._J_needs_update = True
        self._tangent = None
        self._tangent_needs_update = True
        self._cond = None
        self._cond_needs_update = True
        self._J_pred = None
        self._J_pred_needs_update = True
        self._Jpinv = None
        self._Jpinv_needs_update = True

    # Properties
    @property
    def y(self):
        """Current point. (Updated only when corrector step is accepted.)"""
        return self._y

    @y.setter
    def y(self, value):
        """Setter for current point. Sets update flags for derived variables."""
        self._J_needs_update = True
        self._tangent_needs_update = True
        self._cond_needs_update = True
        self._y = value

    @property
    def y_pred(self):
        """Point obtained from Euler prediction, but before corrector loop."""
        return self._y_pred

    @y_pred.setter
    def y_pred(self, value):
        """Setter for predictor point. Sets update flags for derived variables."""
        self._J_pred_needs_update = True
        self._Jpinv_needs_update = True
        self._y_pred = value

    @property
    def J(self):
        """Jacobian, evaluated at y."""
        if self._J_needs_update:
            self._J = self.J_func(self.y)
            self._J_needs_update = False
        return self._J

    @J.setter
    def J(self, value):
        self._J_needs_update = False
        self._J = value

    @property
    def tangent(self):
        """Tangent of implicit curve via QR decomposition of Jacobian.
        Evaluated at y. Normalized to unit length.
        """
        if self._tangent_needs_update:
            Q, R = np.linalg.qr(self.J.transpose(), mode="complete")
            self._tangent = Q[:, -1] * np.sign(R.diagonal().prod())
            self._tangent_needs_update = False
        return self._tangent

    @property
    def cond(self):
        """Condition number of J, evaluated at y."""
        if self._cond_needs_update:
            self._cond = np.linalg.cond(self.J)
            self._cond_needs_update = False
        return self._cond

    @property
    def J_pred(self):
        """Jacobian, evaluated at y_pred."""
        if self._J_pred_needs_update:
            self._J_pred = self.J_func(self.y_pred)
            self._J_pred_needs_update = False
        return self._J_pred

    @property
    def Jpinv(self):
        """Pseudo-inverse of Jacobian, evaluated at y_pred."""
        if self._Jpinv_needs_update:
            self._Jpinv = qr_inv(self.J_pred)
            self._Jpinv_needs_update = False
        return self._Jpinv

    # shorthands
    @property
    def t(self):
        return self._y[-1]

    @property
    def x(self):
        return self._y[:-1]

    def solve(self):
        """Main loop of predictor-corrector steps,
        with step size adaptation between iterations.
        """
        if self.verbose:
            print("=" * 50)
            print("Start homotopy continuation")
        start_time = perf_counter()

        self.converged = False
        self.failed = False

        while not self.converged:
            self.iteration += 1

            self.predict()
            self.correct()
            if self.corrector_success:
                self.check_convergence()

            # Separate if-clause is necessary because check_convergence may rescind
            # corrector success.
            # Not ideal, but not easy to circumvent without additional variable.
            # TODO: come back to this issue and think it through
            if self.corrector_success:
                self.tangent_old = self.tangent
                self.s += np.linalg.norm(self.y - self.y_corr)
                self.y = self.y_corr

                # J at y_corr has already been computed at the end of the corrector step
                # and can be reused
                self.J = self.J_corr

                self.check_bifurcation()

                if self.store_path:
                    self.path.update(
                        y=self.y,
                        s=self.s,
                        cond=self.cond,
                        sign=self.sign,
                        step=self.iteration,
                    )
                if self.verbose:
                    sys.stdout.write(
                        f"\rStep {self.iteration: 5d}:  t ={self.t: 0.4f},  "
                        + f"s ={self.s: 0.2f},  ds ={self.ds: 0.2f},  "
                        + f"cond(J) ={self.cond: 0.0f}     "
                    )
                    sys.stdout.flush()

            if self.converged:
                time_sec = perf_counter() - start_time
                if self.verbose:
                    sys.stdout.write(
                        f"\nStep {self.iteration: 5d}: Continuation successful. "
                        f"Total time elapsed:{timedelta(seconds=int(time_sec))} \n"
                    )
                    sys.stdout.flush()

                return {
                    "success": True,
                    "y": self.y,
                    "s": self.s,
                    "steps": self.iteration,
                    "sign": self.sign,
                    "time": time_sec,
                    "failed": self.failed,
                }

            if self.iteration >= self.max_steps:
                self.failed = "max_steps"

            self.adapt_stepsize()

            if self.failed:
                time_sec = perf_counter() - start_time
                if self.verbose:
                    if self.failed == "predictor":
                        reason = (
                            "Could not find valid predictor: Likely leaving H's domain."
                        )
                    elif self.failed == "max_steps":
                        reason = (
                            "Maximum number of steps reached without convergence. "
                            "(May increase max_steps, then solve() again.)"
                        )
                    elif self.failed == "corrector":
                        reason = "Corrector step failed, and ds is already minimal."
                    sys.stdout.write(f"\nStep {self.iteration: 5d}: {reason} \n")
                    sys.stdout.write(
                        f"\nStep {self.iteration: 5d}: Continuation failed. "
                        f"Total time elapsed:{timedelta(seconds=int(time_sec))} \n"
                    )
                    sys.stdout.flush()

                return {
                    "success": False,
                    "y": self.y,
                    "s": self.s,
                    "steps": self.iteration,
                    "sign": self.sign,
                    "time": time_sec,
                    "failed": self.failed,
                }

    def predict(self):
        """Compute predictor point y_pred, starting at y."""
        self.y_pred = self.y + self.sign * self.ds * self.tangent

        # Check if H contains any NaN at prediction point
        # (which indicates that the predictor step leaves the domain of H).
        # In this case, deflate and try again.
        # If ds is already minimal, stop continuation.
        self.H_pred = self.H_func(self.y_pred)
        if np.isnan(self.H_pred).any():
            if self.ds > self.ds_min:
                self.ds = max(self.ds_defl * self.ds, self.ds_min)
                self.predict()
            else:
                self.failed = "predictor"

    def correct(self):
        """Perform corrector iteration.

        Quasi Newton method:
        Jacobian pseudo-inverse computed once at predictor point,
        not anew at each Newton iteration.
        """
        self.corrector_success = False
        if self.failed:
            return

        corr_dist_old = np.inf
        corr_dist_tot = 0
        self.corr_step = 0

        self.y_corr = self.y_pred.copy()
        H_corr = self.H_pred

        # corrector loop
        while np.max(np.abs(H_corr)) > self.H_tol:
            self.corr_step += 1
            correction = np.dot(self.Jpinv, H_corr)
            self.y_corr = self.y_corr - correction

            corr_dist_step = np.linalg.norm(correction)
            corr_dist_tot += corr_dist_step
            corr_dist = corr_dist_step / max([self.ds, 1])
            # TODO: what is the importance of the denominator above?
            corr_ratio = corr_dist / corr_dist_old
            corr_dist_old = corr_dist

            H_corr = self.H_func(self.y_corr)

            # Check if new corrector point is valid.
            # If H(y) contains any NaN: Correction failed.
            # Reduce stepsize and repeat predictor step.
            if np.isnan(H_corr).any():
                return

            # if corrector step violates parameter restrictions: correction failed.
            # Reduce stepsize and predict again.
            # TODO : sort these error messages / add verbose
            if (
                corr_dist > self.corr_dist_max
                or corr_ratio > self.corr_ratio_max
                or self.corr_step > self.corr_steps_max
            ):
                err_msg = ""
                if corr_dist > self.corr_dist_max:
                    err_msg += (
                        f" corr_dist ={corr_dist: 0.4f} > "
                        f"corr_dist_max ={self.corr_dist_max: 0.4f}; "
                    )
                if corr_ratio > self.corr_ratio_max:
                    err_msg += (
                        f" corr_ratio ={corr_ratio: 0.4f} > "
                        f"corr_ratio_max = {self.corr_ratio_max: 0.4f}; "
                    )
                if self.corr_step > self.corr_steps_max:
                    err_msg += (
                        f" corr_step ={self.corr_step} > "
                        f"corr_steps_max = {self.corr_steps_max}; "
                    )
                cond = np.linalg.cond(self.J_pred)
                err_msg += f" cond(J) = {cond: 0.0f}"
                sys.stdout.write(f"\nStep {self.iteration: 5d}: {err_msg} \n")
                sys.stdout.flush()
                return

        # Corrector loop has converged.
        # Monitor change in determinant of augmented Jacobian.
        # This is an indicator of potential segment jumping.
        # If change is too large, discard new point.
        # Reduce stepsize and repeat predictor step.
        self.J_corr = self.J_func(self.y_corr)
        det_ratio = np.abs(
            np.linalg.det(np.vstack([self.J_corr, self.tangent]))
            / np.linalg.det(np.vstack([self.J, self.tangent]))
        )
        if det_ratio < self.detJ_change_max or det_ratio > 1 / self.detJ_change_max:
            if self.verbose:
                sys.stdout.write(
                    f"\nStep {self.iteration: 5d}: "
                    f"Relative change in augmented determinant: {det_ratio: 0.2f}. "
                    f"Possible segment jump, discarding predictor-corrector step.\n"
                )
                sys.stdout.flush()
            return

        # correction successful.
        self.corrector_success = True

    def check_convergence(self):
        """Check whether convergence is achieved.

        2 possible criteria:
            a) t_target is given and finite.
               Then convergence is achieved if |t_target - t_current| < t_tol.
               [This function also checks if corrector accidentally crossed t_target.
                This should be rare, due to stepsize control.
                If it happens nevertheless, the current step is discarded;
                the algorithm reduces step size and returns to prediction step.]
            b) t_target is inf.
               Then convergence is achieved once all variables (besides t)
               stabilize, and step size is maximal.
        """
        # case a): finite target
        if not np.isinf(self.t_target):
            if np.abs(self.y_corr[-1] - self.t_target) < self.t_tol:
                self.converged = True
            # also check whether t_target was accidentally crossed.
            elif (self.t - self.t_target) * (self.y_corr[-1] - self.t_target) < 0:
                self.corrector_success = False
            # TODO: new flag for this? Or move this to correct()?

        # case b) convergence of x variables
        elif np.isinf(self.t_target):
            if self.ds >= self.ds_max:
                conv_test = (
                    np.max(
                        np.abs(
                            self.x_transformer(self.y[:-1])
                            - self.x_transformer(self.y_corr[:-1])
                        )
                    )
                    / self.ds
                )
                if conv_test < self.x_tol:
                    self.converged = True

    def adapt_stepsize(self):
        """Adapt stepsize at the end of a predictor-corrector cycle:

        Increase ds if:
            - corrector step successful & took less than 10 iterates
        Maintain ds if:
             - corrector step successful, but required 10+ iterates
        Decrease ds if:
            - H could not be evaluated during corrections
              (indicating H's domain has been left)
            - corrector loop fails
              (too many iterates, corrector distance too large,
               or corrector distance increasing during loop)
            - corrector step was successful, but t_target was crossed

        If ds is to be decreased below ds_min,
        continuation is stuck and an error is raised.
        """
        if self.failed:
            return

        if self.corrector_success and self.corr_step < 10:
            self.ds = min(
                self.ds * self.ds_infl,
                self.ds_max,
                np.abs((self.t_target - self.y[-1]) / self.tangent[-1]),
            )
            # TODO: I slightly changed line above,
            # using tangent as denominator rather than previous Dt.
            # Check if OK.
            # Also this might (theoretically at least) lead to a bug where
            # continuation is canceled, if the last expression causes ds < ds_min
            # (possible only if ds_min > t_tol; else convergence happens first.)

        elif not self.corrector_success:
            if self.ds > self.ds_min:
                self.ds = max(self.ds_defl * self.ds, self.ds_min)
            else:
                self.failed = "corrector"

    def check_bifurcation(self):
        """Test whether a bifurcation was crossed and, if so, whether that necessitates
        reversing orientation (=sign).

        After successful prediction/correction step:
        If angle between new and old tangent is close to 180°: perform a sign swap.
        parameter 'bifurc_angle_min' is crucial:
            If too close to 180°, actual bifurcations may be undetected.
            If too far away from 180°, bifurcations may be falsely detected.

        Note:
        Bifurcation detection based on a change in sign of the determinant
        of the augmented Jacobian, as suggested by Allgower/Georg (1990, p. 79)
        did not work well in earlier versions of this program.
        # TODO: Check sign change of determinant again.
        # Does it do the same?
        # See check_convergence.
        """
        if angle(self.tangent_old, self.tangent) > self.bifurc_angle_min:
            if self.verbose:
                sys.stdout.write(
                    f"\nStep {self.iteration: 5d}: Bifurcation point "
                    f"encountered at angle {angle: 0.2f}°. Orientation swapped.\n"
                )
                sys.stdout.flush()
            self.sign = -self.sign

    def set_parameters(self, params: dict = {}, **kwargs):
        """Set multiple parameters at once, given as dictionary and/or as kwargs."""
        # TODO: check inputs against a list to avoid user errors?
        for key in params:
            setattr(self, key, params[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if "ds0" in params or "ds0" in kwargs:
            self.ds = self.ds0

    def check_inputs(self):
        """Check user-provided starting point and homotopy functions."""
        # check y0
        # TODO: might prefer to squeeze here - anything that speaks against that?
        if len(self.y.shape) != 1:
            raise ValueError(
                f'"y0" must be a flat 1D array, but has shape {self.y.shape}.'
            )

        # check H(y0)
        try:
            H0 = self.H_func(self.y)
        except Exception:
            raise ValueError('"H(y0)" cannot be evaluated.')

        if np.max(np.abs(H0)) > self.H_tol:
            raise ValueError('"H(y0)" is not 0.')
        # TODO:

        if np.isnan(H0).any():
            raise ValueError('"H0(y0)" produces NaN.')
        if len(H0.shape) != 1:
            raise ValueError(
                f'"H(y0)" should be a 1D vector, but has shape {H0.shape}.'
            )
        if len(H0) != len(self.y) - 1:
            raise ValueError(
                f'"H(y0)" should have length {len(self.y) - 1}, '
                f"but has length {len(H0)}."
            )

        # check J(y0)
        try:
            J0 = self.J_func(self.y)
        except Exception:
            raise ValueError('"J(y0)" cannot be evaluated.')
        if np.isnan(J0).any():
            raise ValueError('"J(y0)" produces NaN.')
        if len(J0.shape) != 2:
            raise ValueError(
                f'"J(y0)" should be a 2D matrix, ' f"but has shape {J0.shape}."
            )
        if J0.shape != (len(self.y) - 1, len(self.y)):
            raise ValueError(
                f'"J(y0)" should have shape {(len(self.y) - 1, len(self.y))}, '
                f"but has shape {J0.shape}."
            )

        # check x_transformer(self.y[:-1])
        try:
            x0 = self.x_transformer(self.y[:-1])
        except Exception:
            raise ValueError('"x_transformer(y0[:-1])" cannot be evaluated.')
        if np.isnan(x0).any():
            raise ValueError('"x_transformer(y0[:-1])" produces NaN.')
        if len(x0.shape) != 1:
            raise ValueError(
                f'"x_transformer(y0[:-1])" should be a 1D vector, '
                f"but has shape {x0.shape}."
            )

        # Check tansversality at starting point
        orthogonal = np.zeros(len(self.tangent))
        orthogonal[-1] = 1
        tangent_angle = angle(self.tangent, orthogonal)
        if angle(self.tangent, orthogonal) > self.transvers_angle_max:
            print(
                f"Warning: Tangent has angle {tangent_angle:.1f}° "
                f"relative to t-axis. Starting point may violate transversality."
            )

    def set_greedy_sign(self):
        """Set sign so that tangent points towards t_target."""
        self.sign = 1
        t_dir_current = np.sign(self.tangent[-1])
        t_dir_desired = np.sign(self.t_target - self.t)
        if t_dir_current != t_dir_desired:
            self.sign = -1

    def load_state(
        self, y: np.ndarray, sign: int = None, s: int = None, step: int = 0, **kwargs
    ):
        """Load y, and potentially other state variables.
        Prepare to start continuation at this point."""
        self.y = y
        if sign is not None and sign != 0 and not np.isnan(sign):
            self.sign = np.sign(sign)
        else:
            self.set_greedy_sign()

        if s is not None and not np.isnan(s):
            self.s = s
        if step is not None and not np.isnan(step):
            self.iteration = int(step)

        self.ds = self.ds0
        self.check_inputs()

    def load_step(self, step_no):
        """Loads state at step_no from stored path.
        If step_no is not present, the last step preceding it is loaded.
        """
        if self.store_path:
            state = self.path.get_step(step_no)
            if state is not None:
                stored_state, index = self.path.get_step(step_no)
                self.load_state(**stored_state)
                self.path.index = stored_state.index + 1
                print(f"Returning to step {self.iteration}")
        else:
            print(
                "No HomPath assigned. "
                "Assign the path to be loaded from and set store_path to True. "
                "Then try again."
            )


def qr_inv(array):
    """Calculate Moore-Penrose pseudo-inverse of a 2D-array using QR decomposition.
    (Appears to be significantly faster than the equivalent, built-in numpy method
    np.linalg.pinv, which is based on SVD.)
    """
    Q, R = np.linalg.qr(array.transpose(), mode="complete")
    return np.dot(
        Q,
        np.vstack(
            (np.linalg.inv(np.delete(R, -1, axis=0).transpose()), np.zeros(R.shape[1]))
        ),
    )


def angle(vector1, vector2):
    """Calculate the angle between two unit vectors, in degrees."""
    scalar_product = np.clip(
        np.dot(vector1, vector2), -1, 1
    )  # Clipping prevents problems if vectors are colinear.
    return np.arccos(scalar_product) * 180 / np.pi


# %% class for logging homotopy path


class HomPath:
    def __init__(
        self, dim: int, max_steps: int = 500000, x_transformer: callable = lambda x: x
    ):
        """Initialization of path.

        Parameters
        ----------
        dim : int
            Number of variables to be tracked (i.e. len(y))
        max_steps : int, optional
            Maximum number of steps to be tracked, by default 500000.
        x_transformer : callable, optional
            Function to transform x for path length and plotting.
            Defaults to lambda x : x.
        """
        self.max_steps = max_steps
        self.x_transformer = x_transformer
        self.dim = dim

        self.y = np.NaN * np.empty(shape=(max_steps, dim), dtype=np.float64)
        self.s = np.NaN * np.empty(shape=max_steps, dtype=np.float64)
        self.cond = np.NaN * np.empty(shape=max_steps, dtype=np.float64)
        self.sign = np.NaN * np.empty(shape=max_steps, dtype=np.float64)
        self.step = np.NaN * np.empty(shape=max_steps, dtype=np.float64)

        self.index = 0
        self.step_counter = 0

    @property
    def x(self):
        return self.y[:, :-1]

    @property
    def t(self):
        return self.y[:, -1]

    def update(
        self,
        y: np.ndarray,
        s: float = np.NaN,
        cond: float = np.NaN,
        sign: int = np.NaN,
        step: int = None,
    ):

        self.y[self.index] = y
        self.s[self.index] = s
        self.cond[self.index] = cond
        self.sign[self.index] = sign
        if step is not None:
            self.step[self.index] = step
        else:
            self.step[self.index] = self.step_counter
            self.step_counter += 1

        self.index += 1

        if self.index >= self.max_steps:
            self.downsample(10)
            self.index = len(self.s[::10])

    def plot(self, x_name: str = "Variables", max_plotted: int = 100000):
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("Path cannot be plotted: Package matplotlib is required.")
            return None

        if self.index > max_plotted:
            sample_freq = int(np.ceil(max_plotted / self.index))
        else:
            sample_freq = 1
        rows = slice(0, self.index, sample_freq)

        x_plot = self.x_transformer(self.y[rows, :-1])
        t_plot = self.y[rows, -1]
        s_plot = self.s[rows]
        cond_plot = self.cond[rows]
        # sign_plot = self.sign[rows]

        x_plot_min = min([np.amin(x_plot), 0])
        x_plot_max = max([np.amax(x_plot), 1])
        fig = plt.figure(figsize=(10, 7))
        # path length -> homotopy parameter
        ax1 = fig.add_subplot(221)
        ax1.set_title("Homotopy path")
        ax1.set_xlabel(r"path length $s$")
        ax1.set_ylabel(r"homotopy parameter $t$")
        ax1.set_ylim(0, np.max([1, np.amax(t_plot)]))
        ax1.plot(s_plot, t_plot)
        ax1.grid()
        # path length -> variables
        ax2 = fig.add_subplot(222)
        ax2.set_title(fr"{x_name}")
        ax2.set_xlabel(r"path length $s$")
        ax2.set_ylabel(fr"{x_name}")
        ax2.set_ylim(x_plot_min, x_plot_max)
        ax2.plot(s_plot, x_plot)
        ax2.grid()
        # s -> cond(J)
        ax3 = fig.add_subplot(223)
        ax3.set_title("Numerical stability")
        ax3.set_xlabel(r"path length $s$")
        ax3.set_ylabel(r"condition number $cond(J)$")
        ax3.plot(s_plot, cond_plot)
        ax3.grid()
        # t -> y
        ax4 = fig.add_subplot(224)
        ax4.set_title(fr"{x_name} II")
        ax4.set_xlabel(r"homotopy parameter $t$")
        ax4.set_ylabel(fr"{x_name}")
        ax4.set_ylim(x_plot_min, x_plot_max)
        ax4.plot(t_plot, x_plot)
        ax4.grid()
        # ax4 = fig.add_subplot(224)
        # ax4.set_title('Orientation')
        # ax4.set_xlabel(r'path length $s$')
        # ax4.set_ylabel('sign of tangent')
        # ax4.set_ylim(-1.5,1.5)
        # ax4.plot(s_plot, sign_plot)
        # ax4.grid()
        plt.tight_layout()
        plt.show()
        return fig

    def downsample(self, freq):

        cutoff = len(self.s[::freq])

        self.y[:cutoff] = self.y[::freq]
        self.y[cutoff:] = np.NaN
        self.s[:cutoff] = self.s[::freq]
        self.s[cutoff:] = np.NaN
        self.cond[:cutoff] = self.cond[::freq]
        self.cond[cutoff:] = np.NaN
        self.sign[:cutoff] = self.sign[::freq]
        self.sign[cutoff:] = np.NaN
        self.step[:cutoff] = self.step[::freq]
        self.step[cutoff:] = np.NaN

    def cut_nan(self):
        cutoff = np.where(~np.isnan(self.y[:, 0]))[0][-1]
        self.y = self.y[:cutoff]
        self.s = self.s[:cutoff]
        self.cond = self.cond[:cutoff]
        self.sign = self.sign[:cutoff]
        self.step = self.step[:cutoff]

    def get_step(self, step_no: int):
        """Returns data for step_no if possible.
        If x is not present, return the latest step preceeding step_no.
        """
        try:
            index = np.where(self.step == self.step[self.step <= step_no].max())[0]
            return {
                "y": self.y[index].squeeze(),
                "s": self.s[index][0],
                "sign": self.sign[index][0],
                "step": self.step[index][0],
                "index": index,
            }

        except ValueError:
            print(f"Could not find data for any step preceding {step_no}.")
