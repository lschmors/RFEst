import jax.numpy as np
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from sklearn.metrics import mean_squared_error

from .._splines import build_spline_matrix

__all__ = ['splineLNP']


class splineLNP(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=True, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)

    def cost(self, b):

        """
        Negetive Log Likelihood.
        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
        
        def nonlin(x):
            return np.log(1 + np.exp(x)) + 1e-17

        filter_output = nonlin(XS @ b).flatten()
        r = dt * filter_output

        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        if self.lambd:
            l1 = np.sum(np.abs(b))
            l2 = np.sqrt(np.sum(b**2)) 
            neglogli += self.lambd * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
