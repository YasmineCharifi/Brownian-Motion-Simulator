"""
In this module, we will introduce conditional expectation concepts essential for
understanding Brownian motion, martingales, and stochastic filtering.

Key connections to brownian motion:
- Brownian bridge: E[B(t) | B0=x0, B(T)=xt]
- Martingale property: E[B(t)|F_s] = B(s) for s<=t
- Optional stopping and hitting times
- Kalman-Bucy filtering for SDEs
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.interpolate import interp1d
from typing import Literal, Tuple, Optional
import matplotlib.pyplot as plt

class ConditionalExpectationEstimator:
    """
    Estimator for E[X|Y] using various methods:
    this class would provide nonparametric and parametric estimators for
    conditional expectation, fundamental to quantitative finance:
    + Derivatives pricing (risk-neural expectation)
    + Risk Management (expected shortfall)
    + Statistical Arbitrage (factor models)
    """

    def __init__(
            self,
            method: Literal['binning', 'regression', 'kernel'] = 'binning',
            n_bins: int =20,
            bandwidth: Optional[float] = None
    ) -> None:
        self.method = method
        self.n_bins = n_bins
        self.bandwidth = bandwidth
        self.is_fitted = False

        # Fitted attributes
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.bin_centers_: Optional[np.ndarray] = None
        self.bin_means_: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None

    def fit(self, X:ArrayLike, Y:ArrayLike) -> 'ConditionalExpectationEstimator':
        # fit the conditional expectation estimator
        X = np.asarray(X)
        Y = np.asarray(Y)

        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")

        self._X = X
        self._Y = Y

        if self.method == 'binning':
            self._fit_binning(X,Y)
        elif self.method == 'regression':
            self._fit_regression(X,Y)
        elif self.method == 'kernel':
            self._fit_kernel(X,Y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True
        return self

    def _fit_binning(self, X:np.ndarray, Y: np.ndarray) -> None:
        # fit binning estimator
        bin_edges = np.linspace(Y.min(), Y.max(), self.n_bins+1)
        bin_indices = np.clip(np.digitize(Y, bin_edges)-1, 0, self.n_bins-1)

        self.bin_centers_ = (bin_edges[:-1]+ bin_edges[1:])/2
        self.bin_means_ = np.array([
            np.mean(X[bin_indices==b] if np.sum(bin_indices == b)>0 else np.nan 
            for b in range(self.n_bins))
        ])
        self._bin_edges = bin_edges

    def _fit_regression(self, X:np.ndarray, Y: np.ndarray) -> None:
        # fit OLS regression estimator
        self.coef_ = np.cov(X,Y)[0,1]/np.var(Y)
        self.intercept_ = np.mean(X) - np.coef_*np.mean(Y)

    def _fit_kernel(self, X: np.ndarray, Y:np.ndarray) -> None:
        # fit Nadaraya-Watson kernel estimator
        if self.bandwidth is None:
            # Silverman's rule of thumb
            self.bandwidth = 1.06*np.std(Y)*len(Y) **(-1/5)

    def predict(self, y_new:ArrayLike) -> np.ndarray:
        # predicts E[X|Y=y_new] for new values of Y
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before predicting")
        
        y_new = np.asarray(y_new)

        if self.method == 'binning':
            self._predict_binning(y_new)
        elif self.method == 'regression':
            self._predict_regression(y_new)
        elif self.method == 'kernel':
            self._predict_kernel(y_new)

    def _predict_binning(self, y_new: np.ndarray) -> np.ndarray:
        # predicts using binning with interpolation for smoothness
        valid = ~np.isnan(self.bin_means_)
        if np.sum(valid) < 2:
            return np.full_like(y_new, np.nanmean(self.bin_means_), dtype = float)

        # linear interpolation between bin centers
        f = interp1d(
            self.bin_centers_[valid],
            self.bin_means_[valid],
            kind = 'linear',
            bounds_error = False,
            fill_value = (self.bin_means_[valid][0], self.bin_means_[-1])
        )
        return f(y_new)

    def _predict_regression(self, y_new: np.ndarray)  -> np.ndarray:
        # predict using linear regression
        return self.intercept_ + self.coef_*y_new

    def _predict_kernel(self, y_new: np.ndarray) -> np.ndarray:
        # using Nadaraya-Watson kernel regression
        y_new = np.atleast_1d(y_new)
        predictions = np.zeros(len(y_new))

        for i,y in enumerate(y_new):
            # gaussian kernel weights
            weights = np.exp(-0.5*((self._Y-y)/self.bandwidth)**2)
            weights /= weights.sum()
            predictions[i] = np.sum(weights*self._X)

        return predictions

    def score(self, X_true: ArrayLike, Y: ArrayLike) -> float:
        # compute mean squared error of predictions
        X_true = np.asarray(X_true)
        Y = np.asarray(Y)
        predictions = self.predict(Y)
        return np.mean((predictions-X_true))
    
def gaussian_conditional_mean(
        y: ArrayLike,
        mu_x: float,
        mu_y: float,
        sigma_x: float,
        sigma_y: float,
        rho: float
) -> np.ndarray:
    y = np.asarray(y)
    return mu_x + rho*(sigma_x/sigma_y)*(y-mu_y)

def gaussian_conditional_variance(sigma_x: float, rho: float):
    # calculates variance for bivariate gaussian
    return sigma_x**2*(1-rho**2)

def simulate_bivariate_gaussian(
        n:int,
        mu_x: float=0,
        mu_y: float=0,
        sigma_x: float=1,
        sigma_y: float=1,
        rho: float=0.5,
        random_state: Optional[int]=None
) -> Tuple[np.ndarray, np.ndarray]:
    # generate samples from bivariate gaussian distribution
    # uses Cholesky decomposition for numerical stability
    if random_state is not None:
        np.random.seed(random_state)

    # standard normal samples
    Z = np.random.randn(n,2)

    #Cholesky factor of correlation matrix
    X = mu_x + sigma_x*Z[:,0]
    Y = mu_y + sigma_y*(rho*Z[:,0] + np.sqrt(1-rho**2)*Z[:,1])

    return X,Y

def brownian_bridge_expectation(t: ArrayLike, T:float, x0:float=0, xT:float=0) -> np.ndarray:
    # brownian motion that is conditioned on its endpoint
    t = np.asarray(t)
    if np.any(t<0) or np.any(t>T):
        raise ValueError(f"All t values must be in [0,{T}]")
    return x0*(1-t/T)+xT*(t/T)

def brownian_bridge_variance(t: ArrayLike, T:float) -> np.ndarray:
    # computes Var(B(t)|B(0)=0, B(T)=0)=t*(T-t)/T
    # maximum variance reached with t=T/2
    t = np.asarray(t)
    return t*(T-t)/T

def simulate_brownian_bridge(
    t: ArrayLike, T:float, x0: float=0, xT: float=0, n_paths:int=1, random_state: Optional[int]=None
) -> np.ndarray:
    # simulate brownian bridge paths using the conditional distribution
    if random_state is not None:
        np.random.seed(random_state)

    t = np.asarray(t)
    n_times = len(t)

    # mean and variance at each time
    mu = brownian_bridge_expectation(t,T,x0,xT)
    var = brownian_bridge_variance(t,T)

    # generate paths
    paths = np.zeros((n_paths, n_times))
    paths[:,0] = x0
    paths[:,-1] = xT

    # Interior points: sample from conditional distribution
    for i in range(1,n_times-1):
        paths[:,i] = mu[i] + np.sqrt(var[i])*np.random.randn(n_paths)
    
    return paths
    