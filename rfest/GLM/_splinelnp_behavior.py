import jax.numpy as np
import jax.random as random
from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase
from ..utils import build_design_matrix
from ..splines import build_spline_matrix

__all__ = ['splineLNPbehavior']

class splineLNPbehavior(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', nonlinearity='softplus',
                 compute_mle=False, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity

    def forward_pass(self, p, extra=None):
        """
        Model ouput with current estimated parameters.
        """

        XS = self.XS if extra is None else extra['XS']

        if hasattr(self, 'bh_spl'):
            if extra is not None and 'yS' in extra:
                yS = extra['yS']
            else:
                yS = self.yS

        if hasattr(self, 'br_spl'):
            if extra is not None and 'rS' in extra:
                rS = extra['rS']
            else:
                rS = self.rS

        if hasattr(self, 'be_spl'):
            if extra is not None and 'eS' in extra:
                eS = extra['eS']
            else:
                eS = self.eS

        if hasattr(self, 'bo_spl'):
            if extra is not None and 'oS' in extra:
                oS = extra['oS']
            else:
                oS = self.oS

        if self.fit_intercept:
            intercept = p['intercept']
            # run
            if hasattr(self, 'br_spl'):
                intercept_r = p['intercept_r']
            else:
                intercept_r = 1.
            # eye
            if hasattr(self, 'be_spl'):
                intercept_e = p['intercept_e']
            else:
                intercept_e = 1.
            # opto
            if hasattr(self, 'bo_spl'):
                intercept_o = p['intercept_o']
            else:
                intercept_o = 1.
        else:
            if hasattr(self, 'intercept'):
                intercept = self.intercept
                # run
                if hasattr(self, 'br_spl'):
                    intercept_r = self.intercept_r
                else:
                    intercept_r = 1.
                # eye
                if hasattr(self, 'be_spl'):
                    intercept_e = self.intercept_e
                else:
                    intercept_e = 1.
                # opto
                if hasattr(self, 'bo_spl'):
                    intercept_o = self.intercept_o
                else:
                    intercept_o = 1.
            else:
                intercept = 0.
                if hasattr(self, 'br_spl'):
                    intercept_r = 1.
                if hasattr(self, 'be_spl'):
                    intercept_e = 1.
                if hasattr(self, 'bo_spl'):
                    intercept_o = 1.

        if self.fit_R:  # maximum firing rate / scale factor
            R = p['R']
        else:
            if hasattr(self, 'R'):
                R = self.R
            else:
                R = 1.

        if self.fit_nonlinearity:
            nl_params = p['nl_params']
        else:
            if hasattr(self, 'nl_params'):
                nl_params = self.nl_params
            else:
                nl_params = None

        if self.fit_linear_filter:
            filter_output = XS @ p['b']
        else:
            if hasattr(self, 'b_opt'):
                filter_output = XS @ self.b_opt
            else:
                filter_output = XS @ self.b_spl

        if self.fit_history_filter:
            history_output = yS @ p['bh']
        else:
            if hasattr(self, 'bh_opt'):

                history_output = yS @ self.bh_opt
            elif hasattr(self, 'bh_spl'):
                history_output = yS @ self.bh_spl
            else:
                history_output = np.array([0.])

        if self.fit_running_filter:
            running_output = rS @ p['br']
        else:
            if hasattr(self, 'br_opt'):
                running_output = rS @ self.br_opt
            elif hasattr(self, 'br_spl'):
                running_output = rS @ self.br_spl
            else:
                running_output = np.array([0.])

        if self.fit_eye_filter:
            eye_output = eS @ p['be']
        else:
            if hasattr(self, 'be_opt'):
                eye_output = eS @ self.be_opt
            elif hasattr(self, 'be_spl'):
                eye_output = eS @ self.be_spl
            else:
                eye_output = np.array([0.])

        if self.fit_opto_filter:
            opto_output = oS @ p['bo']
        else:
            if hasattr(self, 'bo_opt'):
                opto_output = oS @ self.bo_opt
            elif hasattr(self, 'bo_spl'):
                opto_output = oS @ self.bo_spl
            else:
                opto_output = np.array([0.])

        # r = self.dt * R * self.fnl(filter_output +
        #                           history_output +
        #                           running_output +
        #                           eye_output +
        #                           opto_output +
        #                           intercept,
        #                           nl=self.nonlinearity, params=nl_params).flatten()

        r = self.dt * R * \
            (running_output + intercept_r) * \
            (eye_output + intercept_e) * \
            (opto_output + intercept_o) * \
            self.fnl(
            filter_output +
            history_output +
            intercept,
            nl=self.nonlinearity,
            params=nl_params).flatten()

        return r

    def cost(self, p, extra=None, precomputed=None):

        """
        Negetive Log Likelihood.
        """
        
        y = self.y if extra is None else extra['y']
        r = self.forward_pass(p, extra) if precomputed is None else precomputed 
        r = np.maximum(r, 1e-20) # remove zero to avoid nan in log.
        dt = self.dt

        term0 = - np.log(r / dt) @ y
        term1 = np.sum(r)

        neglogli = term0 + term1
        
        if self.beta and extra is None:
            l1 = np.linalg.norm(p['b'], 1) 
            l2 = np.linalg.norm(p['b'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli


    def initialize_history_filter(self, dims, df, smooth='cr', shift=1):

        """

        Parameters
        ==========

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]

        df : list or array_list
            Number of basis.

        smooth : str
            Type of basis.

        shift : int
            Should be 1 or larger.

        """
        self.shift_h = shift
        y = self.y
        Sh = np.array(build_spline_matrix([dims, ], [df, ], smooth))  # for h
        yh = np.array(build_design_matrix(self.y[:, np.newaxis], Sh.shape[0],
                                          shift=self.shift_h))
        yS = yh @ Sh

        self.yh = np.array(yh)
        self.Sh = Sh  # spline basis for spike-history
        self.yS = yS
        self.bh_spl = np.linalg.solve(yS.T @ yS, yS.T @ y)
        self.h_spl = Sh @ self.bh_spl


    def initialize_running_filter(self, run, dims, df, smooth='cr', shift=1):

        """
        Parameters
        ==========

        run : array_like, shape (n_samples, )
            Recorded running activity.
        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]
        df : int
            Degrees of freedom for spline basis.
        smooth : str
            Specifies the kind of splines. Can be one of the following:
                * 'bs' (B-spline)
                * 'cr' (natural cubic regression spline)
                * 'cc' (cyclic cubic regression spline)
                * 'tp' (truncated Thin Plate regression spline)
        shift : int
            Shift kernel to not predict itself. Should be 1 or larger.

        """

        self.run = run

        Sr = np.array(build_spline_matrix([dims, ], [df, ], smooth))
        rh = np.array(build_design_matrix(self.run[:, np.newaxis], Sr.shape[0], shift=shift))
        rS = rh @ Sr

        self.rh = np.array(rh)
        self.Sr = Sr # spline basis for running filter
        self.rS = rS
        self.br_spl = np.linalg.solve(rS.T @ rS, rS.T @ run)
        self.r_spl = Sr @ self.br_spl


    def initialize_eye_filter(self, eye, dims, df, smooth='cr', shift=1):

        """
        Parameters
        ==========

        eye : array_like, shape (n_samples, )
            Recorded pupil size.
        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]
        df : int
            Degrees of freedom for spline basis.
        smooth : str
            Specifies the kind of splines. Can be one of the following:
                * 'bs' (B-spline)
                * 'cr' (natural cubic regression spline)
                * 'cc' (cyclic cubic regression spline)
                * 'tp' (truncated Thin Plate regression spline)
        shift : int
            Shift kernel to not predict itself. Should be 1 or larger.

        """

        self.eye = eye

        Se = np.array(build_spline_matrix([dims, ], [df, ], smooth))
        eh = np.array(build_design_matrix(self.eye[:, np.newaxis], Se.shape[0], shift=shift))
        eS = eh @ Se

        self.eh = np.array(eh)
        self.Se = Se # spline basis for running filter
        self.eS = eS
        self.be_spl = np.linalg.solve(eS.T @ eS, eS.T @ eye)
        self.e_spl = Se @ self.be_spl


    def initialize_opto_filter(self, opto, dims, df, smooth='cr', shift=1):

        """
        Parameters
        ==========

        opto : array_like, shape (n_samples, )
            Recorded optogenetic stimulation.
            Light is either on or of (array contains only 1.0 or 0.0)
        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]
        df : int
            Degrees of freedom for spline basis.
        smooth : str
            Specifies the kind of splines. Can be one of the following:
                * 'bs' (B-spline)
                * 'cr' (natural cubic regression spline)
                * 'cc' (cyclic cubic regression spline)
                * 'tp' (truncated Thin Plate regression spline)
        shift : int
            Shift kernel to not predict itself. Should be 1 or larger.

        """

        self.opto = opto

        So = np.array(build_spline_matrix([dims, ], [df, ], smooth))
        oh = np.array(build_design_matrix(self.opto[:, np.newaxis], So.shape[0], shift=shift))
        oS = oh @ So

        self.oh = np.array(oh)
        self.So = So # spline basis for opto filter
        self.oS = oS
        self.bo_spl = np.linalg.solve(oS.T @ oS, oS.T @ opto)
        self.o_spl = So @ self.bo_spl


    def fit(self, p0=None, extra=None, initialize='random',
            num_epochs=1, num_iters=3000, metric=None, alpha=1, beta=0.05,
            fit_linear_filter=True, fit_intercept=True, fit_R=True,
            fit_history_filter=False, fit_nonlinearity=False,
            fit_running_filter=False, fit_eye_filter=False, fit_opto_filter=False,
            step_size=1e-2, tolerance=10, verbose=100, random_seed=2046):

        """

        Parameters
        ==========

        p0 : dict
            * 'b': Initial spline coefficients.
            * 'bh': Initial response history filter coefficients
            * 'br': Initial running filter coefficients
            * 'be': Initial eye filter coefficients
            * 'bo': Initial opto filter coefficients

        extra : dict
            Dictionary for test set.
            * 'X': stimulus of test set
            * 'y': response of test set
            * 'run': running of test set (optional, if fit_running_filter=True)
            * 'eye': pupil size of test set (optional, if fit_eye_filter=True)
            * 'opto': optogenetic stimulation of test set (optional, if fit_opto_filter=True)

        initialize : None or str
            Paramteric initalization.
            * if `initialize=None`, `b` will be initialized by b_spl.
            * if `initialize='random'`, `b` will be randomly initialized.

        num_iters : int
            Max number of optimization iterations.

        metric : None or str
            Extra cross-validation metric. Default is `None`. Or
            * 'mse': mean squared error
            * 'r2': R2 score
            * 'corrcoef': Correlation coefficient

        alpha : float, from 0 to 1.
            Elastic net parameter, balance between L1 and L2 regulization.
            * 0.0 -> only L2
            * 1.0 -> only L1

        beta : float
            Elastic net parameter, overall weight of regulization for receptive field.

        step_size : float
            Initial step size for JAX optimizer.

        tolerance : int
            Set early stop tolerance. Optimization stops when cost monotonically
            increases or stop increases for tolerance=n steps.

        verbose: int
            When `verbose=0`, progress is not printed. When `verbose=n`,
            progress will be printed in every n steps.

        """

        self.metric = metric

        self.alpha = alpha
        self.beta = beta  # elastic net parameter - global penalty weight for linear filter
        self.num_iters = num_iters

        self.fit_R = fit_R
        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept
        self.fit_running_filter = fit_running_filter
        self.fit_eye_filter = fit_eye_filter
        self.fit_opto_filter = fit_opto_filter

        # initial parameters

        if p0 is None:
            p0 = {}

        dict_keys = p0.keys()
        if 'b' not in dict_keys:
            if initialize is None:
                p0.update({'b': self.b_spl})
            else:
                if initialize == 'random':
                    key = random.PRNGKey(random_seed)
                    b0 = 0.01 * random.normal(key, shape=(self.n_b * self.n_c,)).flatten()
                    p0.update({'b': b0})

        if 'intercept' not in dict_keys:
            p0.update({'intercept': np.array([0.])})
            if hasattr(self, 'br_spl'):
                p0.update({'intercept_r': np.array([0.])})
            if hasattr(self, 'be_spl'):
                p0.update({'intercept_e': np.array([0.])})
            if hasattr(self, 'bo_spl'):
                p0.update({'intercept_o': np.array([0.])})

        if 'R' not in dict_keys:
            p0.update({'R': np.array([1.])})

        if 'bh' not in dict_keys:
            if hasattr(self, 'bh_spl'):
                p0.update({'bh': self.bh_spl})
            else:
                p0.update({'bh': None})
        if 'br' not in dict_keys:
            if hasattr(self, 'br_spl'):
                p0.update({'br': self.br_spl})
            else:
                p0.update({'br': None})
        if 'be' not in dict_keys:
            if hasattr(self, 'be_spl'):
                p0.update({'be': self.be_spl})
            else:
                p0.update({'be': None})
        if 'bo' not in dict_keys:
            if hasattr(self, 'bo_spl'):
                p0.update({'bo': self.bo_spl})
            else:
                p0.update({'bo': None})

        if 'nl_params' not in dict_keys:
            if hasattr(self, 'nl_params'):
                p0.update({'nl_params': self.nl_params})
            else:
                p0.update({'nl_params': None})

        if extra is not None:

            if self.n_c > 1:
                XS_ext = np.dstack(
                    [extra['X'][:, :, i] @ self.S for i in range(self.n_c)]).reshape(
                    extra['X'].shape[0], -1)
                extra.update({'XS': XS_ext})
            else:
                extra.update({'XS': extra['X'] @ self.S})

            if hasattr(self, 'h_spl'):
                yh_ext = np.array(
                    build_design_matrix(extra['y'][:, np.newaxis], self.Sh.shape[0],
                                        shift=self.shift_h))
                yS_ext = yh_ext @ self.Sh
                extra.update({'yS': yS_ext})
            if hasattr(self, 'r_spl'):
                r_ext = np.array(build_design_matrix(extra['run'][:, np.newaxis],
                                                     self.Sr.shape[0], shift=1))
                rS_ext = r_ext @ self.Sr
                extra.update({'rS': rS_ext})
            if hasattr(self, 'e_spl'):
                e_ext = np.array(build_design_matrix(extra['eye'][:, np.newaxis],
                                                     self.Se.shape[0], shift=1))
                eS_ext = e_ext @ self.Se
                extra.update({'eS': eS_ext})
            if hasattr(self, 'o_spl'):
                o_ext = np.array(build_design_matrix(extra['opto'][:, np.newaxis],
                                                     self.So.shape[0], shift=1))
                oS_ext = o_ext @ self.So
                extra.update({'oS': oS_ext})

            extra = {key: np.array(extra[key]) for key in extra.keys()}

            self.extra = extra  # store for cross-validation

        # store optimized parameters
        self.p0 = p0
        self.p_opt = self.optimize_params(p0, extra, num_epochs, num_iters, metric, step_size,
                                          tolerance, verbose)
        self.R = self.p_opt['R'] if fit_R else np.array([1.])

        if fit_linear_filter:
            self.b_opt = self.p_opt['b']  # optimized RF basis coefficients
            if self.n_c > 1:
                self.w_opt = self.S @ self.b_opt.reshape(self.n_b, self.n_c)
            else:
                self.w_opt = self.S @ self.b_opt  # optimized RF

        if fit_history_filter:
            self.bh_opt = self.p_opt['bh']
            self.h_opt = self.Sh @ self.bh_opt

        if fit_running_filter:
            self.br_opt = self.p_opt['br']
            self.r_opt = self.Sr @ self.br_opt

        if fit_eye_filter:
            self.be_opt = self.p_opt['be']
            self.e_opt = self.Se @ self.be_opt

        if fit_opto_filter:
            self.bo_opt = self.p_opt['bo']
            self.o_opt = self.So @ self.bo_opt

        if fit_nonlinearity:
            self.nl_params_opt = self.p_opt['nl_params']

        if fit_intercept:
            self.intercept = self.p_opt['intercept']
            if fit_running_filter:
                self.intercept_r = self.p_opt['intercept_r']
            if fit_eye_filter:
                self.intercept_e = self.p_opt['intercept_e']
            if fit_opto_filter:
                self.intercept_o = self.p_opt['intercept_o']


    def predict(self, X, y=None, run=None, eye=None, opto=None, p=None):

        """

        Parameters
        ==========

        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : None or array_like, shape (n_samples, )
            Recorded response. Needed when post-spike filter is fitted.

        run : None or array_like, shape (n_samples, )
            Recorded running activity. Needed when running filter is fitted.

        eye : None or array_like, shape (n_samples, )
            Recorded pupil size. Needed when eye filter is fitted.

        opto : None or array_like, shape (n_samples, )
            Optogenetic light stimulation. Needed when eye filter is fitted.

        p : None or dict
            Model parameters. Only needed if model performance is monitored
            during training.

        """

        if self.n_c > 1:
            XS = np.dstack([X[:, :, i] @ self.S for i in range(self.n_c)]).reshape(X.shape[0],
                                                                                   -1)
        else:
            XS = X @ self.S

        extra = {'X': X, 'XS': XS, 'y': y}

        if hasattr(self, 'h_spl'):

            if y is None:
                raise ValueError('`y` is needed for calculating response history.')

            yh = np.array(
                build_design_matrix(extra['y'][:, np.newaxis], self.Sh.shape[0],
                                    shift=self.shift_h))
            yS = yh @ self.Sh
            extra.update({'yS': yS})

        if hasattr(self, 'r_spl'):
            if (run is None) & (self.fit_running_filter is True):
                raise ValueError('`run` is needed for calculating response.')
            extra['run'] = run
            rh = np.array(build_design_matrix(extra['run'][:, np.newaxis], self.Sr.shape[0],
                                              shift=1))
            rS = rh @ self.Sr
            extra.update({'rS': rS})

        if hasattr(self, 'e_spl'):
            if (eye is None) & (self.fit_eye_filter is True):
                raise ValueError('`eye` is needed for calculating response.')
            extra['eye'] = eye
            eh = np.array(build_design_matrix(extra['eye'][:, np.newaxis], self.Se.shape[0],
                                              shift=1))
            eS = eh @ self.Se
            extra.update({'eS': eS})

        if hasattr(self, 'o_spl'):
            if (opto is None) & (self.fit_opto_filter is True):
                raise ValueError('`opto` is needed for calculating response.')
            extra['opto'] = opto
            oh = np.array(build_design_matrix(extra['opto'][:, np.newaxis], self.So.shape[0],
                                              shift=1))
            oS = oh @ self.So
            extra.update({'oS': oS})

        params = self.p_opt if p is None else p
        y_pred = self.forward_pass(params, extra=extra)

        return y_pred
