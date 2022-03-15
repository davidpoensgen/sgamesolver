import numpy as np
import time
from datetime import timedelta
import sys


class HomCont:
    """Class to perform homotopy continuation to solve a nonlinear system of equations: F(x) = H(x, t_target) = 0.

        Given:  1) System of equations H(x,t) = 0 with homotopy parameter t.
                2) Known solution x0 at t0, i.e. H(x0, t0) = 0.

        Wanted: A solution x* at t_target, i.e. H(x*, t_target) = 0.

        Idea:   Start at y0 = (x0,t0) and trace implied path up to y* = (x*, t_target).

        x is a vector of dimension N
        t is a homotopy parameter
        y is shorthand for vector (x, t)

        Main inputs
        -----------
        H : callable
            Homotopy function: R^(N+1) -> R^N
        y0 : np.ndarray
            The starting point for homotopy continuation.
            Must be 1D array.
            Must (approximately) solve the system H(y0) = 0.
            The homotopy parameter t is stored in the last entry.
            The variables of interest x are stored in the other entries.
        J : callable, optional
            Jacobian matrix of H: R^(N+1) -> R^N x R^(N+1)
            If not provided by user, a finite difference approximation is used (requires package numdifftools).

        t_target : float, optional
            Target value of the homotopy parameter, by default np.inf.
            If np.inf, iteration continues until all components of x converge.
            (Convergence is checked using values transformed by x_transformer - see below.)
        sign: int, +1, -1, or None
            Orientation in which the path is traced.
            If None (default), program will choose orientation so that tracing starts towards t_target.
            (Usually this should simply be left at None, unless the user has a specific reason to set a
              certain orientation.)
        max_steps : float, optional
            Maximum number of predictor-corrector iterations, by default np.inf.
        x_transformer : callable, optional
            Transformation of x to check for convergence, by default lambda x : x.
            (TODO: comment?)
        verbose : int, optional
            Determines how much feedback is displayed during continuation:
            0 : silent, no reports at all.
            1 : only start and end of continuation are reported. This is the default.
            2 : current progress is reported continuously. also reports special occurrences, e.g. orientation reversals.
            3 : additional reports for parameter tuning or debugging. Includes failed corrector loops,
                discarded steps due to potential segment jumping.

        store_path: bool, optional
            If True, the path traversed during continuation is stored as HomPath, accessible as this.path.
            Allows to graph the path afterwards, and also to restart continuation at previous points.
            Will slow down solution somewhat, and increases memory footprint.


        parameters : dict
            Collection of parameters for path tracing, which will override the defaults.
            May alternatively be passed as kwargs.
            Method .set_parameters() allows to adjust parameters after construction,
            again passed as dict or kwargs.

            The parameters are:
            ----------
            x_tol : float
                Continuation is considered to have converged successfully once max(|x_new-x_old|) / ds < x_tol,
                i.e. x has stabilized. x is transformed by x_transformer for this criterion, if provided.
                Active only if t_target = np.inf. Defaults to 1e-7.
            t_tol : float
                Continuation is considered to have converged successfully once |t - t_target| < t_tol.
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
                Maximum relative change of determinant of augmented Jacobian between consecutive
                predictor-corrector steps: Steps are discarded unless
                detJ_change_max < |detJ_new|/|det_j_old| < 1/detJ_change_max.
                (Large relative changes in augmented determinant indicate potential segment jumping.)
                Defaults to 0.5.
            bifurc_angle_min : float
                Minimum angle (in degrees) between two consecutive predictor
                tangents to be considered a bifurcation, defaults to 177.5.
                If a bifurcations is crossed, path orientation is swapped.
    """

    def __init__(self,
                 H: callable,
                 y0: np.ndarray,
                 J: callable = None,
                 t_target: float = np.inf,
                 max_steps: float = np.inf,
                 sign: int = None,
                 x_transformer: callable = lambda x: x,
                 verbose: int = 1,
                 store_path: bool = False,
                 parameters: dict = {},
                 **kwargs):

        self.H_func = H
        if J is not None:
            self.J_func = J
        else:
            try:
                import numdifftools as nd
            except ModuleNotFoundError:
                raise ModuleNotFoundError('If J is not provided by user, package numdifftools is required.')
            else:
                self.J_func = nd.Jacobian(H)
        self.y = y0.squeeze()

        self.t_target = t_target
        self.max_steps = max_steps
        self.x_transformer = x_transformer
        self.verbose = verbose

        # TODO: keep?
        self.normalize_J = False

        # set default parameters
        self.x_tol = 1e-7
        self.t_tol = 1e-7
        self.H_tol = 1e-7
        self.ds0 = 0.01
        self.ds_infl = 1.2
        self.ds_defl = 0.5
        self.ds_min = 1e-9
        self.ds_max = 1000
        self.corr_steps_max = 20
        self.corr_dist_max = 0.3
        self.corr_ratio_max = 0.3
        self.detJ_change_max = 0.5
        self.bifurc_angle_min = 177.5

        if sign is not None and sign != 0:
            self.sign = np.sign(sign)
        else:
            self.set_greedy_sign()

        self.tangent_old = self.tangent
        self.ds = self.ds0

        self.store_path = store_path
        if store_path:
            self.path = HomPath(dim=len(self.y), x_transformer=self.x_transformer)
            self.path.update(y=self.y, s=0, cond=self.cond, sign=self.sign, step=0, ds=self.ds)

        # attributes to be used later
        self.step = 0
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

        # overwrite defaults with user-provided parameters if present
        self.set_parameters(parameters, **kwargs)

        self.check_inputs()

    # Properties
    @property
    def y(self):
        """Current point. (Updated only when corrector step is accepted.)"""
        return self._y

    @y.setter
    def y(self, value):
        """Set current point and update flags for derived variables."""
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
        """Set predictor point and update flags for derived variables."""
        self._J_pred_needs_update = True
        self._Jpinv_needs_update = True
        self._y_pred = value

    @property
    def J(self):
        """Jacobian, evaluated at y."""
        if self._J_needs_update:
            self._J = self.J_func(self.y)
            if self.normalize_J:
                self._J /= np.abs(self._J).max(axis=1, keepdims=True)
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
            Q, R = np.linalg.qr(self.J.transpose(), mode='complete')
            self._tangent = Q[:, -1] * np.sign(R.diagonal()).prod()
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
            if self.normalize_J:
                self._J_pred /= np.abs(self._J_pred).max(axis=1, keepdims=True)
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

    def solve(self):  # sourcery no-metrics
        """Main loop of predictor-corrector steps,
        with step size adaptation between iterations.
        """
        if self.verbose >= 1:
            print('=' * 50)
            print('Start homotopy continuation')
        start_time = time.perf_counter()

        self.converged = False
        self.failed = False

        while not self.converged:
            self.step += 1

            self.predict()
            self.correct()
            if self.corrector_success:
                self.check_convergence()

            # separate if-clause is necessary, because check_convergence may
            # rescind corrector success (if t_target is overshot).
            if self.corrector_success:
                self.tangent_old = self.tangent
                self.s += np.linalg.norm(self.y - self.y_corr)
                self.y = self.y_corr

                # J at y_corr has already been computed and can be reused:
                self.J = self.J_corr

                self.check_bifurcation()

                if self.verbose >= 2:
                    sys.stdout.write(f'\rStep {self.step:5d}: t = {self.t:#.4g},  '
                                     f's = {self.s:#.4g},  ds = {self.ds:#.4g},  '
                                     f'cond(J) = {self.cond:#.4g}      ')
                    sys.stdout.flush()

            if self.converged:
                time_sec = time.perf_counter() - start_time
                if self.verbose >= 1:
                    sys.stdout.write(f'\nStep {self.step:5d}: Continuation successful. '
                                     f'Total time elapsed:{timedelta(seconds=int(time_sec))} \n')
                    sys.stdout.flush()

                return {'success': True,
                        'y': self.y,
                        's': self.s,
                        'steps': self.step,
                        'sign': self.sign,
                        'time': time_sec,
                        'failure reason': False,
                        }

            if self.step >= self.max_steps:
                self.failed = 'max_steps'

            self.adapt_stepsize()

            if self.corrector_success and self.store_path:
                self.path.update(y=self.y, s=self.s, cond=self.cond, sign=self.sign, step=self.step, ds=self.ds)

            if self.failed:
                time_sec = time.perf_counter() - start_time
                if self.verbose >= 1:
                    if self.failed == 'predictor':
                        reason = 'Could not find valid predictor: Likely hit a boundary of H\'s domain.'
                    elif self.failed == 'max_steps':
                        reason = 'Maximum number of steps reached without convergence. ' \
                                 '(May increase max_steps, then solve() again.)'
                    elif self.failed == 'corrector':
                        reason = 'Corrector step failed, and ds is already minimal.'
                    sys.stdout.write(f'\nStep {self.step:5d}: {reason} \n')
                    sys.stdout.write(f'Step {self.step:5d}: Continuation failed. '
                                     f'Total time elapsed:{timedelta(seconds=int(time_sec))} \n')
                    sys.stdout.flush()

                return {'success': False,
                        'y': self.y,
                        's': self.s,
                        'steps': self.step,
                        'sign': self.sign,
                        'time': time_sec,
                        'failure reason': self.failed,
                        }

    def predict(self):
        """Compute predictor point y_pred, starting at y."""
        self.y_pred = self.y + self.sign * self.ds * self.tangent

        # Check if H contains any NaN at prediction point (which indicates that the predictor step leaves the
        # domain of H). In this case, deflate and try again. If ds is already minimal, stop continuation.
        self.H_pred = self.H_func(self.y_pred)
        if np.isnan(self.H_pred).any():
            if self.ds > self.ds_min:
                self.ds = max(self.ds_defl*self.ds, self.ds_min)
                self.predict()
            else:
                self.failed = 'predictor'

    def correct(self):  # sourcery no-metrics
        """Perform corrector iteration.

        Method is quasi-Newton: Jacobian pseudo-inverse is computed once at
        predictor point, not anew at each Newton iteration.
        """
        self.corrector_success = False
        if self.failed:
            return

        corr_dist_old = np.inf
        self.corr_step = 0

        self.y_corr = self.y_pred
        H_corr = self.H_pred

        # corrector loop
        while np.max(np.abs(H_corr)) > self.H_tol:
            self.corr_step += 1
            correction = np.dot(self.Jpinv, H_corr)
            self.y_corr = self.y_corr - correction

            corr_dist = np.linalg.norm(correction)
            corr_ratio = corr_dist / corr_dist_old
            corr_dist_old = corr_dist

            H_corr = self.H_func(self.y_corr)

            # If H(y) contains any NaN (corrector step has left domain of H):
            # Correction failed, reduce stepsize and repeat predictor step.
            if np.isnan(H_corr).any():
                return

            # If corrector step violates restrictions given by parameters:
            # Correction failed, reduce stepsize and predict again.
            # Note: corr_dist_max has to be relaxed for large ds: thus, * max(ds, 1)
            if (corr_dist > self.corr_dist_max * max(self.ds, 1)
                    or corr_ratio > self.corr_ratio_max
                    or self.corr_step > self.corr_steps_max):
                if self.verbose >= 3:
                    err_msg = 'Corrector loop failed.'
                    if corr_dist > self.corr_dist_max * max(self.ds, 1):
                        err_msg += f' corr_dist = {corr_dist/max(self.ds, 1):0.4f} (max: {self.corr_dist_max:0.4f});'
                    if corr_ratio > self.corr_ratio_max:
                        err_msg += f' corr_ratio = {corr_ratio:0.4f} (max: {self.corr_ratio_max:0.4f});'
                    if self.corr_step > self.corr_steps_max:
                        err_msg += f' corr_step = {self.corr_step} (max: {self.corr_steps_max});'
                    cond = np.linalg.cond(self.J_pred)
                    err_msg += f' cond(J_pred) = {cond:#.4g}'
                    sys.stdout.write(f'\nStep {self.step:5d}: {err_msg} \n')
                    sys.stdout.flush()

                return

        # Corrector loop has converged.
        # Monitor change in determinant of augmented Jacobian.
        # This is an indicator of potential segment jumping.
        # If change is too large, discard new point.
        # Reduce stepsize and repeat predictor step.
        # Use log determinant to deal with large matrices.
        self.J_corr = self.J_func(self.y_corr)

        if self.normalize_J:
            self.J_corr /= np.abs(self.J_corr).max(axis=1, keepdims=True)

        old_log_det = np.linalg.slogdet(np.vstack([self.J, self.tangent]))[1]
        new_log_det = np.linalg.slogdet(np.vstack([self.J_corr, self.tangent]))[1]
        log_det_diff = new_log_det - old_log_det
        # det_ratio = np.exp(new_log_det - old_log_det)

        # TODO: remove old version
        # det_ratio = np.abs(np.linalg.det(np.vstack([self.J_corr, self.tangent])) /
        #                    np.linalg.det(np.vstack([self.J, self.tangent])))
        # if det_ratio > self.detJ_change_max or det_ratio < 1/self.detJ_change_max:

        if log_det_diff > np.log(self.detJ_change_max) or log_det_diff < np.log(1/self.detJ_change_max):
            if self.verbose >= 3:
                sys.stdout.write(f'\nStep {self.step:5d}: Possible segment jump, discarding step. '
                                 # f'Ratio of augmented determinants: det(J_new)/det(J_old) = {det_ratio:0.2f}\n')
                                 f'Augmented determinants: logdet(J_new) - logdet(J_old) = {log_det_diff:0.2f}\n')
                sys.stdout.flush()
            return

        self.corrector_success = True

    def check_convergence(self):
        """Check whether convergence is achieved.

       2 possible criteria:
           a) t_target is given and finite.
              Then convergence is achieved if |t_target - t_current| < t_tol.
              [This function also checks if corrector accidentally crossed t_target.
               This should be rare, due to stepsize control. In that case, the current
               step is discarded, the algorithm reduces ds is and returns to the prediction step.]
           b) t_target is inf.
              Then convergence is achieved once all variables (besides t)
              have stabilized, and step size is maximal.
              [If x_transformer is specified, the transformed variables are used for this criterion.]
        """
        # Case a): t_target is finite
        if not np.isinf(self.t_target):
            if np.abs(self.y_corr[-1] - self.t_target) < self.t_tol:
                self.converged = True
            # also check whether t_target was accidentally crossed.
            elif (self.t - self.t_target) * (self.y_corr[-1] - self.t_target) < 0:
                self.corrector_success = False

        # Case b): t_target is infinite
        elif np.isinf(self.t_target):
            if self.ds >= self.ds_max:
                conv_test = (np.max(np.abs(self.x_transformer(self.y[:-1]) - self.x_transformer(self.y_corr[:-1])))
                             / self.ds)
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
             (indicates leaving H's domain)
           - corrector loop fails (too many iterates, corrector distance
             too large, or corrector distance increasing during loop)
           - corrector step was successful, but t_target was crossed
       If ds is to be decreased below ds_min, continuation is failed.

       If t_target is finite, stepsize is capped so that the predictor will not cross t_target
        """
        if self.failed:
            return

        if self.corrector_success and self.corr_step < 10:
            self.ds = min(self.ds * self.ds_infl, self.ds_max)

        elif not self.corrector_success:
            if self.ds > self.ds_min:
                self.ds = max(self.ds_defl * self.ds, self.ds_min)
            else:
                self.failed = "corrector"

        if not np.isinf(self.t_target):
            try:
                cap = np.abs((self.t_target - self.y[-1]) / self.tangent[-1])
                self.ds = min(self.ds, cap)
            except ZeroDivisionError:
                pass

    def check_bifurcation(self):
        """Test whether a bifurcation was crossed that necessitates reversing orientation ("sign").

       After successful prediction/correction step:
       If angle between new and old tangent is close to 180°: perform a sign swap.
       parameter 'bifurc_angle_min' is crucial:
           If too close to 180°, actual bifurcations may be undetected.
           If too far away from 180°, bifurcations may be falsely detected.

       Note:
       A possible alternative, bifurcation detection based on a sign change of the determinant
       of the augmented Jacobian (as suggested by Allgower/Georg, 1990, p. 79) seems to miss some
       points where a change of orientation is necessary. This is possibly because it is only guaranteed
       to detect simple bifurcations, and does not necessarily detect higher order bifurcations.
        """
        if angle(self.tangent_old, self.tangent) > self.bifurc_angle_min:
            if self.verbose >= 2:
                sys.stdout.write(f'\nStep {self.step:5d}: Bifurcation point encountered '
                                 f'at angle {angle(self.tangent_old, self.tangent):0.2f}°. Orientation swapped.\n')
                sys.stdout.flush()
            self.sign = -self.sign

    def set_parameters(self, params: dict = None, **kwargs):
        """Set multiple parameters at once, given as dictionary and/or as kwargs."""
        params = params or {}
        inputs = {**params, **kwargs}
        for key, value in inputs.items():
            if not hasattr(self, key):
                print(f'Warning: "{key}" is not a valid parameter.')
            setattr(self, key, value)
        if 'ds0' in params or 'ds0' in kwargs:
            self.ds = self.ds0

    def check_inputs(self):
        """Check user-provided starting point and homotopy functions."""
        # check y0
        if len(self.y.shape) != 1:
            raise ValueError(f'"y0" must be a flat 1D array, but has shape {self.y.shape}.')

        # check H(y0)
        try:
            H0 = self.H_func(self.y)
        except Exception:
            raise ValueError('"H(y0)" cannot be evaluated.')

        if np.isnan(H0).any():
            raise ValueError('"H0(y0)" produces NaN.')
        if len(H0.shape) != 1:
            raise ValueError(f'"H(y0)" should be a 1D vector, but has shape {H0.shape}.')
        if len(H0) != len(self.y) - 1:
            raise ValueError(f'"H(y0)" should have length {len(self.y) - 1}, '
                             f'but has length {len(H0)}.')

        if np.max(np.abs(H0)) > self.H_tol:
            print(f'Warning: "H(y0)" is not 0 (max deviation: {np.max(np.abs(H0))}).\n'
                  '   Solution might still be possible (because the first corrector step may fix this issue).\n'
                  '   However, it is advised to start with a better approximation for the starting point.')

        # check J(y0)
        try:
            J0 = self.J_func(self.y)
        except Exception:
            raise ValueError('"J(y0)" cannot be evaluated.')
        if np.isnan(J0).any():
            raise ValueError('"J(y0)" produces NaN.')
        if len(J0.shape) != 2:
            raise ValueError(f'"J(y0)" should be a 2D matrix, but has shape {J0.shape}.')
        if J0.shape != (len(self.y) - 1, len(self.y)):
            raise ValueError(f'"J(y0)" should have shape {(len(self.y) - 1, len(self.y))}, but has shape {J0.shape}.')

        # check x_transformer(self.y[:-1])
        try:
            x0 = self.x_transformer(self.y[:-1])
        except Exception:
            raise ValueError('"x_transformer(y0[:-1])" cannot be evaluated.')
        if np.isnan(x0).any():
            raise ValueError('"x_transformer(y0[:-1])" produces NaN.')
        if len(x0.shape) != 1:
            raise ValueError(f'"x_transformer(y0[:-1])" should be a 1D vector, but has shape {x0.shape}.')

        # Check transversality at starting point
        t_axis = np.zeros(len(self.tangent))
        t_axis[-1] = 1
        tangent_angle = angle(self.tangent, t_axis)
        if abs(tangent_angle) < 2.5:
            print(f'Warning: Tangent has angle {tangent_angle:.1f}° '
                  'relative to t-axis. Starting point may violate transversality.')

    def set_greedy_sign(self):
        """Set sign so that continuation starts towards t_target."""
        self.sign = 1
        t_dir_current = np.sign(self.tangent[-1])
        t_dir_desired = np.sign(self.t_target - self.t)
        if t_dir_current != t_dir_desired:
            self.sign = -1

    def load_state(self, y: np.ndarray, sign: int = None,  s: float = None, step: int = 0, ds: float = None, **kwargs):
        """Load y, and potentially other state variables. Prepare to start continuation at this point."""
        self.y = y
        if sign is None or sign == 0 or np.isnan(sign):
            self.set_greedy_sign()
        else:
            self.sign = np.sign(sign)

        if s is not None and not np.isnan(s):
            self.s = s
        if step is not None and not np.isnan(step):
            self.step = int(step)

        if ds is not None:
            self.ds = ds
        else:
            self.ds = self.ds0

        self.check_inputs()

    def return_to_step(self, step_no):
        """Loads state at step_no from stored path, or if not present, the last step preceding it."""
        if self.store_path:
            state = self.path.get_step(step_no)
            if state is not None:
                self.load_state(**state)
                self.path.index = state['index'] + 1
                print(f'Returning to step {self.step}.')
        else:
            print('Path not stored and no HomPath assigned.')

    def save_file(self, filename, overwrite: bool = False):
        """Save current state of the solver to a file.

        Allows to re-start continuation from the current state later on.
        Note: H, J and parameters are not saved. User should make sure these can be recreated.
        Path history (HomPath) is not saved either.
        """
        import os
        import json
        if os.path.isfile(filename) and not overwrite:
            answer = input(f'"{filename}" already exists. Overwrite content [y/N]?')
            if answer == '' or answer[0].lower() != 'y':
                print('Saving to file canceled.')
                return

        with open(filename, 'w') as file:
            state = {'description': f'HomCont state saved on {time.ctime()}.',
                     'step': self.step,
                     's': self.s,
                     'sign': self.sign,
                     'ds': self.ds,
                     'y': self.y.tolist(),
                     }
            json.dump(state, file, indent=4)
            print(f'Current state saved as {filename}.')

    def load_file(self, filename):
        """Load solver state from a file created by save_file()."""
        import os
        import json
        if not os.path.isfile(filename):
            print(f'{filename} not found.')
            return
        with open(filename) as file:
            state = json.load(file)
            state['y'] = np.array(state['y'])
            self.load_state(**state)
            print(f'State successfully loaded from {filename}.')


def qr_inv(array):
    """Calculate Moore-Penrose pseudo-inverse of a 2D-array using QR decomposition.

    Note: Appears to be significantly faster than the equivalent, built-in numpy method
    np.linalg.pinv, which is based on SVD.
    """
    Q, R = np.linalg.qr(array.transpose(), mode='complete')
    return np.dot(Q, np.vstack((np.linalg.inv(np.delete(R, -1, axis=0).transpose()), np.zeros(R.shape[1]))))


def angle(vector1, vector2):
    """Calculate the angle between two unit vectors, in degrees."""
    scalar_product = np.clip(np.dot(vector1, vector2), -1, 1)
    # Clipping prevents problems if vectors are collinear.
    return np.arccos(scalar_product) * 180 / np.pi


class HomPath:
    """Container to store path data.

    Parameters
    ----------
    dim : int
        Number of variables to be tracked (i.e. len(y))
    max_steps : int, optional
        Maximum number of steps to be tracked, by default 10000.
    x_transformer : callable, optional
        Function to transform x for plotting, by default lambda x : x.
    """

    def __init__(self, dim: int, max_steps: int = 10000, x_transformer: callable = lambda x: x):
        self.max_steps = max_steps
        self.x_transformer = x_transformer
        self.dim = dim

        self.y = np.nan * np.empty(shape=(max_steps, dim), dtype=np.float64)
        self.s = np.nan * np.empty(shape=max_steps, dtype=np.float64)
        self.cond = np.nan * np.empty(shape=max_steps, dtype=np.float64)
        self.sign = np.nan * np.empty(shape=max_steps, dtype=np.float64)
        self.step = np.nan * np.empty(shape=max_steps, dtype=np.float64)
        self.ds = np.nan * np.empty(shape=max_steps, dtype=np.float64)

        self.index = 0

    def update(self, y: np.ndarray, s: float, cond: float, sign: int, step: int, ds: float):

        self.y[self.index] = y
        self.s[self.index] = s
        self.cond[self.index] = cond
        self.sign[self.index] = sign
        self.step[self.index] = step
        self.ds[self.index] = ds

        self.index += 1
        if self.index >= self.max_steps:
            self.downsample(10)

    def plot(self, x_name: str = 'Variables', max_plotted: int = 1000):
        """Plot path."""
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print('Path cannot be plotted: Package matplotlib is required.')
            return None

        if self.index > max_plotted:
            sample_freq = int(np.ceil(max_plotted/self.index))
        else:
            sample_freq = 1
        rows = slice(0, self.index, sample_freq)

        x_plot = self.x_transformer(self.y[rows, :-1])
        t_plot = self.y[rows, -1]
        s_plot = self.s[rows]
        cond_plot = self.cond[rows]

        x_plot_min = min([np.amin(x_plot), 0])
        x_plot_max = max([np.amax(x_plot), 1])
        fig = plt.figure(figsize=(10, 7))
        # path length -> homotopy parameter
        ax1 = fig.add_subplot(221)
        ax1.set_title('Homotopy path')
        ax1.set_xlabel(r'path length $s$')
        ax1.set_ylabel(r'homotopy parameter $t$')
        ax1.set_ylim(0, np.max([1, np.amax(t_plot)]))
        ax1.plot(s_plot, t_plot)
        ax1.grid()
        # path length -> variables
        ax2 = fig.add_subplot(222)
        ax2.set_title(fr'{x_name}')
        ax2.set_xlabel(r'path length $s$')
        ax2.set_ylabel(fr'{x_name}')
        ax2.set_ylim(x_plot_min, x_plot_max)
        ax2.plot(s_plot, x_plot)
        ax2.grid()
        # s -> cond(J)
        ax3 = fig.add_subplot(223)
        ax3.set_title('Numerical stability')
        ax3.set_xlabel(r'path length $s$')
        ax3.set_ylabel(r'condition number $cond(J)$')
        ax3.plot(s_plot, cond_plot)
        ax3.grid()
        # t -> y
        ax4 = fig.add_subplot(224)
        ax4.set_title(fr'{x_name} II')
        ax4.set_xlabel(r'homotopy parameter $t$')
        ax4.set_ylabel(fr'{x_name}')
        ax4.set_ylim(x_plot_min, x_plot_max)
        ax4.plot(t_plot, x_plot)
        ax4.grid()
        # alternatively: sign on axis 4
        # sign_plot = self.sign[rows]
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
        for variable in [self.y, self.s, self.cond, self.sign, self.step, self.ds]:
            variable[:cutoff] = variable[::freq]
            variable[cutoff:] = np.NaN
        self.index = cutoff

    def get_step(self, step_no: int):
        """Returns data for step_no if possible.
        If step_no is not present, the last step preceding it is returned instead.
        """
        try:
            index = np.where(self.step == self.step[self.step <= step_no].max())[0]
            state = {'y': self.y[index].squeeze(),
                     's': self.s[index][0],
                     'sign': self.sign[index][0],
                     'step': self.step[index][0],
                     'index': int(index),
                     'ds': self.ds[index][0]
                     }
            return state

        except ValueError:
            print(f'Could not find data for any step preceding {step_no}.')
