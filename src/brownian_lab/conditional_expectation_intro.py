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
from pathlib import Path

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
            np.mean(X[bin_indices==b]) if np.sum(bin_indices == b)>0 else np.nan 
            for b in range(self.n_bins)
        ])
        self._bin_edges = bin_edges

    def _fit_regression(self, X:np.ndarray, Y: np.ndarray) -> None:
        # fit OLS regression estimator
        self.coef_ = np.cov(X,Y)[0,1]/np.var(Y)
        self.intercept_ = np.mean(X) - self.coef_*np.mean(Y)

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
            return self._predict_binning(y_new)
        elif self.method == 'regression':
            return self._predict_regression(y_new)
        elif self.method == 'kernel':
            return self._predict_kernel(y_new)

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
            fill_value = (self.bin_means_[valid][0], self.bin_means_[valid][-1])
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
        return np.mean((predictions-X_true)**2)
    
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

def demonstrate_conditional_expectation(n_samples:int=5000, rho:float=0.7, n_bins:int=20, figsize: Tuple[int,int]=(14,10)) -> dict:
    # this function serves as : a visual teaching tool and a validation of estimation methods
    print("="*60)
    print("Conditional Expectation Demonstration")
    print("="*60)

    # generate data
    X, Y = simulate_bivariate_gaussian(n_samples, rho=rho, random_state=42)

    # fit estimators
    binning_est = ConditionalExpectationEstimator(method='binning', n_bins=n_bins)
    regression_est = ConditionalExpectationEstimator(method='regression')
    kernel_est = ConditionalExpectationEstimator(method='kernel')

    binning_est.fit(X,Y)
    regression_est.fit(X,Y)
    kernel_est.fit(X,Y)

    # create prediction grid
    y_grid = np.linspace(Y.min(), Y.max(), 200)

    # predictions
    pred_binning = binning_est.predict(y_grid)
    pred_regression = regression_est.predict(y_grid)
    pred_kernel = kernel_est.predict(y_grid)
    pred_true = gaussian_conditional_mean(y_grid,0,0,1,1,rho)

    # compute MSEs
    true_at_Y = gaussian_conditional_mean(Y,0,0,1,1,rho)
    mse_binning = np.mean((binning_est.predict(Y) - true_at_Y)**2)
    mse_regression = np.mean((regression_est.predict(Y)-true_at_Y)**2)
    mse_kernel = np.mean((kernel_est.predict(Y)-true_at_Y)**2)

    # plot
    fig, axes = plt.subplots(2,2, figsize=figsize)

    # plot 1: data with all estimators
    ax1 = axes[0,0]
    idx = np.random.choice(n_samples, min(1500, n_samples), replace=False)
    ax1.scatter(Y[idx], X[idx], alpha=0.2, s=10, c='gray', label='Data')
    ax1.plot(y_grid, pred_true, 'g-', lw=3, label='True E[X|Y]')
    ax1.plot(y_grid, pred_regression, 'b--', lw=2, label='regression')
    ax1.plot(y_grid, pred_binning, 'r', lw=2, label='binning')
    ax1.plot(y_grid, pred_kernel, 'm-', lw=2, label='kernel')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')
    ax1.set_title(f'Conditional Expectation Estimation (rho={rho})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # plot 2: Estimation errors
    ax2 = axes[0,1]
    ax2.plot(y_grid, pred_regression-pred_true, 'b-', lw=2, label=f'regression (MSE={mse_regression:.2e})')
    ax2.plot(y_grid, pred_binning-pred_true, 'b-', lw=2, label=f'binning (MSE={mse_binning:.2e})')
    ax2.plot(y_grid, pred_kernel-pred_true, 'b-', lw=2, label=f'kernel (MSE={mse_kernel:.2e})')
    ax2.axhline(y=0, color='green', linestyle='--', lw=2)
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Error')
    ax2.set_title('Estimation Error vs True E[X|Y]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # plot 3: Brownian Bridge
    ax3 = axes[1,0]
    T=1.0
    t=np.linspace(0,T,100)
    bridge_paths = simulate_brownian_bridge(t,T,x0=0,xT=0,n_paths=20, random_state=42)
    bridge_mean = brownian_bridge_expectation(t,T,0,0)
    bridge_std = np.sqrt(brownian_bridge_variance(t,T))

    for path in bridge_paths:
        ax3.plot(t, path, 'b-', alpha=0.3, lw=0.8)
    ax3.plot(t, bridge_mean, 'r-', lw=3, label='E[B(t)|B(0)=0, B(T)=0]')
    ax3.fill_between(t, bridge_mean-2*bridge_std, bridge_mean+2*bridge_std, alpha=0.2, color='red', label='+2*std band')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # plot 4: Bridge variance
    ax4 = axes[1,1]
    ax4.plot(t, bridge_std**2, 'purple', lw=3)
    ax4.axvline(x=T/2, color='red', linestyle='--', label=f'Max variance at t=T/2')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Var(B(t)|B(0)=0, B(T)=0)')
    ax4.set_title('Brownian Bridge Conditional Variance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.tight_layout()

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(figures_dir / "conditional_expec_demo.png",dpi=150,bbox_inches="tight")
    plt.show()

    # print summary
    print(f"\nSample size: {n_samples:,}")
    print(f"Correlation: rho = {rho}")
    print("\nMean Squared Errors(vs True E[X|Y])")
    print(f"Regression: {mse_regression:.6f}")
    print(f"Binning: {mse_binning:.6f}")
    print(f"Kernel: {mse_kernel:.6f}")
    print(f"\nIrreducible variance:{gaussian_conditional_variance(1, rho):.4f}")

    return{
        'estimators':{
            'binning': binning_est,
            'regression': regression_est,
            'kernel': kernel_est,
        },
        'mse':{
            'binning': mse_binning,
            'regression': mse_regression,
            'kernel': mse_kernel,
        },
        'data':{'X':X, 'Y':Y},
        'grid': y_grid
    }
if __name__=="__main__":
    # run demo
    results= demonstrate_conditional_expectation(
        n_samples= 5000,
        rho=0.7,
        n_bins=20
    )

    print("\n"+"="*60)
    print("BROWNIAN BRIDGE EAMPLE")
    print("="*60)

    t_test = np.array([0.0,0.25,0.5,0.75,1.0])
    T=1.0

    print("Brownian Bridge: B(0)=0 and B(1)=1")
    print('-'*60)
    for ti in t_test:
        mean = brownian_bridge_expectation(ti,T,x0=0,xT=1)
        var = brownian_bridge_variance(ti,T)
        print(f"t={ti:.2f}:E[B(t)|..]={mean:.3f}, Var={var:.4f}")

    print("\n Module Demo Complete")