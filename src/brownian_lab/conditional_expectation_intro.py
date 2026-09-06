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