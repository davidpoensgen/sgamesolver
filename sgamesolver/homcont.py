import numpy as np
import time
from datetime import timedelta
import sys
import os
import json


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
        If not provided by user, a finite difference approximation is used (requires package numdifftools; likely
        orders of magnitude slower).

    Convergence criteria
    -----------
    t_target : float, optional
    convergence_tol: float, optional
    x_tol: float, optional
    distance_function: callable, optional

        The solver allows 2 possible modes to determine convergence:
        a) if t_target is a real number, the solver will attempt to find a solution to H(x, t_target) = 0.
           Specifically, the solver stops once solution to H(x,t) = 0 is found for which |t-t_target| < t_tol.
        b) if t_target is np.inf (the default), the solver will let t increase without bounds, but continuously
           monitor whether all other variables, i.e. x, have converged. Concretely, the convergence criterion is
           max(|x_new - x_old|) / |t_new - t_old| < x_tol.
           If desired, one can pass a distance_function to the solver to be used for this criterion instead.
           This function should take 2 arguments, the vectors y_new and y_old, and return a non-negative
           real number. Convergence then requires distance_function(y_new, y_old) < x_tol.
           A possible use case: H takes variables as input that are a transformation of those one is actually
           interested in, e.g. logarithms; then one can use a custom distance_function to monitor whether the
           original, non-transformed variables have converged.
           Note that it is also possible to define convergence criteria in this manner that are not actually
           distances. An example from the economic context might be to have distance_function calculate the
           epsilon-deviation from optimal behavior for y_new, and terminate once this is lower than x_tol.

    sign: int, +1, -1, or None
        Orientation in which the path is traced.
        If None (default), program will choose orientation so that tracing starts towards t_target.
        (Usually this should simply be left at None, unless the user has a specific reason to set a
          certain orientation.)
    max_steps : int, optional
        Maximum number of predictor-corrector iterations, by default np.inf.
    verbose : int, optional
        Determines how much feedback is displayed during continuation:
        0 : Silent, no reports at all.
        1 : Current progress is reported continuously. This is the default.
        2 : Also reports special occurrences, e.g. orientation reversals.
        3 : Additional reports for parameter tuning or debugging. Includes failed corrector loops,
            discarded steps due to potential segment jumping.


    parameters : dict
        Collection of parameters for path tracing, which will override the defaults.
        May alternatively be passed as kwargs.
        Method .set_parameters() allows to adjust parameters after construction,
        again passed as dict or kwargs.

        The parameters are:
        ----------
        x_tol : float  # TODO: rename to distance_tol? merge parameters x_tol, t_tol to convergence_tol
                       #TODO:  , since it's only ever one that matters?
            Continuation is considered to have converged successfully once max(|x_new-x_old|) / |t_new-t_old| < x_tol,
            i.e. x has stabilized. If a distance_function is provided, it is used to calculate the distance instead.
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
            Convergence criterion used for corrector step: max(|H(y_corr)|) < H_tol.
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
            Minimum angle (in degrees) between two consecutive predictor tangents to be considered a bifurcation,
            defaults to 177.5. If a bifurcations is crossed, path orientation is swapped.
    """

    def __init__(self,
                 H: callable,
                 y0: np.ndarray,
                 J: callable = None,
                 t_target: float = np.inf,
                 max_steps: int = np.inf,
                 sign: int = None,
                 distance_function: callable = None,
                 verbose: int = 1,
                 parameters: dict = None,
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
        if distance_function is not None:
            self.distance = distance_function
        else:
            self.distance = self.distance_function

        self.verbose = verbose

        # set default parameters
        self.x_tol = 1e-7
        self.t_tol = 1e-7
        self.H_tol = 1e-7
        self.ds0 = 0.01
        self.ds_infl = 1.2
        self.ds_infl_max_corr_steps = 9
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

        # attributes to be used later
        self.start_time = None
        self.step = 0
        self.s = 0.0
        self.corrector_success = False
        self.corr_fail_dist = False
        self.corr_fail_ratio = False
        self.corr_fail_steps = False
        self.converged = False
        self.det_ratio = None

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

        self.store_path = False
        self.path = None
        self.store_cond = False
        self.test_segment_jumping = False

        self.debug = False

    # Properties: These mainly serve to cache the results of potentially expensive function calls
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
            self._J_pred_needs_update = False
        return self._J_pred

    @property
    def Jpinv(self):
        """Pseudo-inverse of Jacobian, evaluated at y_pred."""
        if self._Jpinv_needs_update:
            self._Jpinv = qr_inv(self.J_pred)
            self._Jpinv_needs_update = False
        return self._Jpinv

    # shorthand for t = y[-1]
    @property
    def t(self):
        return self._y[-1]

    def start(self):
        """Main loop of predictor-corrector steps, with step size adaptation between iterations."""
        if self.verbose >= 1:
            print('=' * 50)
            print('Start homotopy continuation')
        self.start_time = time.perf_counter()

        self.converged = False

        while not self.converged:
            # try-except block: allows sub-functions to exit the main loop by raising ContinuationFailed
            try:
                self.step += 1

                self.predict()
                self.correct()
                if self.debug:
                    self.debug.update()

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

                    if self.verbose >= 1:
                        self._report_step()

                if self.converged:
                    return self._report_result()

                self.adapt_stepsize()

                if self.store_path and self.corrector_success:
                    self.path.update()

                if self.step >= self.max_steps:
                    raise ContinuationFailed('max_steps')

            except ContinuationFailed as exception:
                return self._report_result(exception=exception)

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
                raise ContinuationFailed('predictor')

    def correct(self):
        """Perform corrector iteration.

        Method is quasi-Newton: Jacobian pseudo-inverse is computed once at
        predictor point, not anew at each Newton iteration.
        """
        self.corrector_success = False
        self.corr_fail_dist = False
        self.corr_fail_ratio = False
        self.corr_fail_steps = False
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

            # If corrector step violates any restriction given by parameters:
            # Correction failed, reduce stepsize and predict again.
            # Note: corr_dist_max has to be relaxed for large ds: thus, * max(ds, 1)
            # TODO: current implementation of corr_steps_max is not sensible: checks violation after performing step?
            self.corr_fail_dist = corr_dist > self.corr_dist_max * max(self.ds, 1)
            self.corr_fail_ratio = corr_ratio > self.corr_ratio_max
            self.corr_fail_steps = self.corr_step > self.corr_steps_max
            if self.corr_fail_dist or self.corr_fail_ratio or self.corr_fail_steps:
                if self.verbose >= 3:
                    err_msg = 'Corrector loop failed.'
                    if self.corr_fail_dist:
                        err_msg += f' corr_dist = {corr_dist/max(self.ds, 1):0.2f} (max: {self.corr_dist_max:0.2f});'
                    if self.corr_fail_ratio:
                        err_msg += f' corr_ratio = {corr_ratio:0.2f} (max: {self.corr_ratio_max:0.2f});'
                    if self.corr_fail_steps:
                        err_msg += f' corr_step = {self.corr_step} (max: {self.corr_steps_max});'
                    cond = np.linalg.cond(self.J_pred)
                    err_msg += f' cond(J_pred) = {cond:#.4g}'
                    self._report_step()
                    print(f'\nStep {self.step:5d}: {err_msg}')

                return

            # If corrector has not failed: get new H
            H_corr = self.H_func(self.y_corr)

            # If H(y) contains any NaN (corrector step has left domain of H):
            # Correction failed, reduce stepsize and repeat predictor step.
            if np.isnan(H_corr).any():
                return

        # Corrector loop has converged.
        self.J_corr = self.J_func(self.y_corr)

        if self.test_segment_jumping:
            # Optional test for large relative changes in augmented determinant - a potential indicator for segment
            # jumping (see Choi et al. 1995). Uses slogdet to avoid overflows for large systems.
            # If a potential jump is detected, the step is discarded and ds decreased.
            old_log_det = np.linalg.slogdet(np.vstack([self.J, self.tangent]))[1]
            new_log_det = np.linalg.slogdet(np.vstack([self.J_corr, self.tangent]))[1]
            log_det_diff = np.abs(new_log_det - old_log_det)
            self.det_ratio = np.exp(log_det_diff)
            if log_det_diff > np.abs(np.log(self.detJ_change_max)):
                if self.verbose >= 3:
                    self._report_step()
                    print(f'\nStep {self.step:5d}: Possible segment jump, discarding step. Ratio of augmented'
                          f' determinants: |det(J_new) / det(J_old)| = {self.det_ratio:0.2f}')
                return

        self.corrector_success = True

    def check_convergence(self):
        """Check whether convergence is achieved.

       2 possible criteria:
           a) t_target is a finite real number.
              Then convergence is achieved if |t_target - t_current| < t_tol.
              [This function also checks if corrector accidentally crossed t_target.
               This should be rare, due to stepsize control. In that case, the current
               step is discarded, the algorithm reduces ds is and returns to the prediction step.]
           b) t_target is inf or -inf.
              Then convergence is achieved once all variables (besides t) have stabilized, and step size is maximal,
              i.e. distance(y_new-y_old) < x_tol. By default, distance is measured using the method
              distance_function; it is possible to pass an alternative function to be used instead.
        """
        # Case a): t_target is finite
        if not np.isinf(self.t_target):
            if np.abs(self.y_corr[-1] - self.t_target) < self.t_tol:
                self.converged = True
            # otherwise, check whether t_target was accidentally crossed.
            elif (self.t - self.t_target) * (self.y_corr[-1] - self.t_target) < 0:
                self.corrector_success = False

        # Case b): t_target is infinite
        elif np.isinf(self.t_target):
            if self.ds >= self.ds_max:
                if self.distance(y_new=self.y_corr, y_old=self.y) < self.x_tol:
                    self.converged = True

    def adapt_stepsize(self):
        """Adapt stepsize at the end of a predictor-corrector cycle:
        Increase ds if:
           - corrector step successful & took less than ds_infl_max_corr_steps iterates (= 10 by default)
        Maintain ds if:
            - corrector step successful, but required 10+ iterates
        Decrease ds if:
           - corrector loop fails (too many iterates, corrector distance
             too large, or corrector distance increasing during loop)
           - H could not be evaluated during corrections
             (indicates leaving H's domain)
           - corrector step was successful, but t_target was crossed
        If ds is to be decreased below ds_min, continuation is failed.

        If t_target is finite, stepsize is capped so that the predictor will not cross t_target.
        """

        if self.corrector_success and self.corr_step <= self.ds_infl_max_corr_steps:
            self.ds = min(self.ds * self.ds_infl, self.ds_max)

        elif not self.corrector_success:
            if self.ds > self.ds_min:
                self.ds = max(self.ds_defl * self.ds, self.ds_min)
            else:
                raise ContinuationFailed("corrector")

        if not np.isinf(self.t_target):
            try:
                cap = (self.t_target - self.y[-1]) / (self.tangent[-1] * self.sign)
                # step length has to be capped only if current movement is towards t_target:
                if cap > 0:
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
                self._report_step()
                print(f'\nStep {self.step:5d}: Bifurcation point encountered at '
                      f'angle {angle(self.tangent_old, self.tangent):0.2f}°. Orientation swapped.')
            self.sign = -self.sign

    def set_parameters(self, params: dict = None, **kwargs):
        """Set multiple parameters at once, given as dictionary and/or as kwargs."""
        params = params or {}
        inputs = {**params, **kwargs}
        for key, value in inputs.items():
            if not hasattr(self, key):
                print(f'Warning: "{key}" is not a valid parameter.')
            setattr(self, key, value)
        if 'ds0' in inputs:
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

        # Check transversality at starting point
        if self.verbose >= 2:
            t_axis = np.zeros_like(self.tangent)
            t_axis[-1] = 1
            tangent_angle = angle(self.tangent, t_axis)
            if abs(90 - tangent_angle) < 2.5:
                print(f'Warning: Tangent has angle {tangent_angle:.1f}° '
                      'relative to t-axis. Starting point may violate transversality.')

    def set_greedy_sign(self):
        """Set sign so that continuation starts towards t_target."""
        self.sign = 1
        t_direction_current = np.sign(self.tangent[-1])
        t_direction_desired = np.sign(self.t_target - self.t)
        if t_direction_current != t_direction_desired:
            self.sign = -1

    @staticmethod
    def distance_function(y_new, y_old):
        """Calculate maximum difference in y[:-1], normalized by difference in t.
        Possible convergence criterion.
        """
        abs_difference = np.abs(y_new - y_old)
        return np.max(abs_difference[:-1]) / abs_difference[-1]

    def _report_result(self, exception=None) -> dict:
        """Return a dictionary with continuation result; print message if verbose."""
        time_sec = time.perf_counter() - self.start_time

        if exception is None:
            failure_reason = None
            success = True
        else:
            failure_reason = exception.reason
            success = False

        if self.verbose >= 1:
            if success:
                print(f'\nStep {self.step:5d}: Continuation successful. '
                      f'Total time elapsed:{timedelta(seconds=int(time_sec))}')
            else:
                print(f'\nStep {self.step:5d}: Failure reason: {exception.message}')
                print(f'Step {self.step:5d}: Continuation failed. '
                      f'Total time elapsed:{timedelta(seconds=int(time_sec))}')
            print('End homotopy continuation')
            print('=' * 50)

        return {'success': success,
                'y': self.y,
                's': self.s,
                'steps': self.step,
                'sign': self.sign,
                'time': time_sec,
                'failure reason': failure_reason,
                }

    def _report_step(self):
        output = f'\rStep {self.step:5d}: t = {self.t:#6.4g}, s = {self.s:#6.4g}, ds = {self.ds:#6.4g}'
        if self.store_cond:
            output += f', Cond(J) = {self.cond:#6.4g}'
        print(output, end='', flush=True)

    def _load_state(self, y: np.ndarray, sign: int = None, s: float = None, step: int = 0, ds: float = None, **kwargs):
        """Load y and other state variables. Prepare to start continuation at this point."""
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
                self._load_state(**state)
                self.path.index = state['index'] + 1
                print(f'Returning to step {self.step}.')
        else:
            print('There is no stored path.')

    def save_file(self, filename, overwrite=False):
        """Save current state of the solver to a file.

        Allows to re-start continuation from the current state later on.
        Note: H, J and parameters are not saved. User should make sure these can be recreated.
        Path history (HomPath) is not saved either.
        """
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
                     'y': self.y.tolist()}
            json.dump(state, file, indent=4)
            print(f'Current state saved as {filename}.')

    def load_file(self, filename):
        """Load solver state from a file created by save_file()."""
        if not os.path.isfile(filename):
            print(f'{filename} not found.')
            return
        with open(filename) as file:
            state = json.load(file)
            state['y'] = np.array(state['y'])
            self._load_state(**state)
            print(f'State successfully loaded from {filename}.')

    def start_storing_path(self, max_steps: int = 1000):
        """Initialises path storing. This will allow to return to earlier steps,
        or plot the progression of variables along the path."""
        self.store_path = True
        if not self.path:
            self.path = HomPath(solver=self, max_steps=max_steps)
            self.path.update()

    def start_debug_log(self):
        """Initializes a debug log to store information on corrector steps."""
        self.debug = DebugLog(self)


class ContinuationFailed(Exception):
    """Exception raised by subfunctions to exit main predictor-corrector-loop."""
    def __init__(self, reason):
        self.reason = reason
        if reason == 'predictor':
            self.message = 'Could not find valid predictor: Likely hit a boundary of H\'s domain.'
        elif reason == 'max_steps':
            self.message = 'Maximum number of steps reached without convergence. ' \
                           '(To continue, increase max_steps, then start again.)'
        elif reason == 'corrector':
            self.message = 'Corrector step failed, and ds is already minimal.'
        else:
            self.message = "Something unexpected happened."

        super().__init__(self, self.message)

    def __str__(self):
        return self.message


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
    """Container to store path data for the specified solver instance."""

    def __init__(self, solver: HomCont, max_steps: int = 1000):
        self.max_steps = max_steps
        self.solver = solver

        self.y = np.nan * np.empty(shape=(max_steps, solver.y.shape[0]))
        self.s = np.nan * np.empty(shape=max_steps)
        self.cond = np.nan * np.empty(shape=max_steps)
        self.sign = np.nan * np.empty(shape=max_steps)
        self.step = np.nan * np.empty(shape=max_steps)
        self.ds = np.nan * np.empty(shape=max_steps)

        self.index = 0
        self.downsample_frequency = 10

    def update(self):
        """Store current state of solver."""
        self.y[self.index] = self.solver.y
        self.s[self.index] = self.solver.s
        if self.solver.store_cond:
            self.cond[self.index] = self.solver.cond
        self.sign[self.index] = self.solver.sign
        self.step[self.index] = self.solver.step
        self.ds[self.index] = self.solver.ds

        self.index += 1
        if self.index >= self.max_steps:
            self.downsample(self.downsample_frequency)

    def plot(self, max_plotted: int = 1000, y_indices: list = None):
        """Plot path.
        If a list or array of y_indices is given, only these are plotted.
        To plot a range of indices, you can use np.arange, e.g. y_indices = np.arange(10, 20).
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print('Missing the the python package matplotlib. Please install to plot.')
            return

        if self.index > max_plotted:
            sample_freq = int(np.ceil(max_plotted/self.index))
        else:
            sample_freq = 1
        rows = slice(0, self.index, sample_freq)

        if y_indices is None:
            x_plot = self.y[rows, :-1]
        else:
            x_plot = self.y[rows, y_indices]

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
        ax2.set_title(fr'Variables in y')
        ax2.set_xlabel(r'path length $s$')
        ax2.set_ylabel(fr'$y_i$')
        ax2.set_ylim(x_plot_min, x_plot_max)
        ax2.plot(s_plot, x_plot)
        ax2.grid()
        # s -> cond(J): plot only if cond has been stored.
        if np.invert(np.isnan(cond_plot)).any():
            ax3 = fig.add_subplot(223)
            ax3.set_title('Numerical stability')
            ax3.set_xlabel(r'path length $s$')
            ax3.set_ylabel(r'condition number $cond(J)$')
            ax3.plot(s_plot, cond_plot)
            ax3.grid()
        else:
            # alternatively: ds on axis 3
            ds_plot = self.ds[rows]
            ax3 = fig.add_subplot(223)
            ax3.set_title('step size')
            ax3.set_xlabel(r'path length $s$')
            ax3.set_ylabel('ds')
            ax3.plot(s_plot, ds_plot)
            ax3.grid()
        # t -> y
        ax4 = fig.add_subplot(224)
        ax4.set_title(fr'Variables in y II')
        ax4.set_xlabel(r'homotopy parameter $t$')
        ax4.set_ylabel(fr'$y_i$')
        ax4.set_ylim(x_plot_min, x_plot_max)
        ax4.plot(t_plot, x_plot)
        ax4.grid()
        plt.tight_layout()
        plt.show()
        return fig

    def downsample(self, frequency):
        """Free up space by keeping only a subset of existing data with specified sampling frequency."""
        cutoff = len(self.s[::frequency])
        for variable in [self.y, self.s, self.cond, self.sign, self.step, self.ds]:
            variable[:cutoff] = variable[::frequency]
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


class DebugLog:
    """Log to collect data on corrector steps for parameter tuning and debugging."""
    def __init__(self, solver: HomCont):
        self.solver = solver
        self.index = 0
        self.data = np.zeros((6, 1000))

    @property
    def step(self):
        return self.data[0, :self.index]

    @property
    def corrector_steps(self):
        return self.data[1, :self.index]

    @property
    def corrector_fail_steps(self):
        return self.data[2, :self.index]

    @property
    def corrector_fail_dist(self):
        return self.data[3, :self.index]

    @property
    def corrector_fail_ratio(self):
        return self.data[4, :self.index]

    @property
    def det_ratio(self):
        return self.data[5, :self.index]

    def update(self):
        self.data[0, self.index] = self.solver.step
        self.data[1, self.index] = self.solver.corr_step
        self.data[2, self.index] = self.solver.corr_fail_steps
        self.data[3, self.index] = self.solver.corr_fail_dist
        self.data[4, self.index] = self.solver.corr_fail_ratio
        if self.solver.test_segment_jumping:
            self.data[5, self.index] = self.solver.det_ratio

        self.index += 1
        if self.index >= self.data.shape[1]:
            self.data.resize((self.data.shape[0], self.data.shape[1] + 1000), refcheck=False)

    def summarize(self):
        header = "steps |"
        counts = "total |"
        ratio = "ratio |"
        dist = "dist  |"
        for i in range(self.solver.corr_steps_max + 1):
            count = (self.corrector_steps == i).sum()
            ratio_count = ((self.corrector_fail_ratio > 0) * (self.corrector_steps == i)).sum()
            dist_count = ((self.corrector_fail_dist > 0) * (self.corrector_steps == i)).sum()
            width = 1 + max(2, len(str(count)))
            header += str(i).rjust(width) + "|"
            counts += str(count).rjust(width) + "|"
            ratio += str(ratio_count).rjust(width) + "|"
            dist += str(dist_count).rjust(width) + "|"
        failure_count = (self.corrector_fail_steps + self.corrector_fail_ratio + self.corrector_fail_dist > 0).sum()
        summary = (
            f'~~~~~~~~~~ debug summary ~~~~~~~~~~\n'
            f'Total steps: {self.index}\n'
            f'Number of corrector steps (average: {np.average(self.corrector_steps):.2f})\n'
            f'{header}\n'
            f'{counts}\n'
            f'{ratio}\n'
            f'{dist}\n'
            f'Total failed corrector loops: {failure_count:.0f}. Failure reasons: \n'
            f'- Max steps: {self.corrector_fail_steps.sum():.0f}\n'
            f'- Ratio: {self.corrector_fail_ratio.sum():.0f}\n'
            f'- Distance: {self.corrector_fail_dist.sum():.0f}'
        )
        print(summary)
