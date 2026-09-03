"""
Conditional Probability in Brownian Motion

We will simulate and analyze conditional distributions of Brownian Motion,
including Brownian Bridges and path interpolation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional

##################################################################################################
# Core simulation functions
##################################################################################################

def simulate_brownian_motion(
        T : float,
        n_steps: int,
        n_paths: int=1,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    # returning times of shape :(n_steps+1) and paths with shape(n_paths, n_steps+1) 
    if seed is not None:
        np.random.seed(seed)
    dt = T/n_steps
    times = np.linspace(0, T, n_steps+1)

    increments = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

    paths = np.zeros((n_paths, n_steps+1))
    paths[:,1:] = np.cumsum(increments, axis=1)

    return times, paths

def simulate_brownian_bridge(
        T: float,
        n_steps: int,
        a: float,
        b: float,
        n_paths: int=1,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    # simulating briwnian bridges from a to b over [0,T]
    # for the mean, we use linear interpolation between endpoints
    # and for the variance, the maximum is at the midpoint which means
    # the most uncertainty is in the middle
    times, bm_paths = simulate_brownian_motion(T, n_steps, n_paths, seed)
    # B^br(t) = B(t) - (t/T)*B(T) + (t/T)*b + (1-t/T)*a
    ratios = times/T
    B_T = bm_paths[:, -1:] #shape (n_paths, 1)
    bridge_paths = (bm_paths - ratios* B_T + ratios*b + (1-ratios)*a)

    return times, bridge_paths

##################################################################################################
# Distribution functions
##################################################################################################

def conditional_distribution_future(s: float, t: float, x_s: float)-> Tuple[float, float]:
    # distribution of B(t) given B(s); t>s
    # B(t)|B(s) = x_s ~ N(x_s, t-s)
    if t <= s:
        raise ValueError(f"Require t>s, got t={t}, s={s}")
    return x_s, t-s

def brownian_bridge_distribution(t:float, T:float, a:float, b:float) -> Tuple[float, float]:
    # B^br(t) ~ N(a+(t/T)(b-a), t(T-t)/T)
    mean = a + (t/T)*(b-a)
    variance = t*(T-t)/T
    return mean, variance

def conditional_interpolation(s:float, t:float, u:float, x_s:float, x_t:float) -> Tuple[float, float]:
    # it can be seen as a mini brownian between s and t
    if not(s<u<t):
        raise ValueError(f"Requires s<u<t, got s={s}, u={u} and t={t}")
    mean = x_s +(u-s)/(t-s)*(x_t-x_s)
    variance = (u-s)*(t-u)/(t-s)
    return mean, variance

def prob_max_exceeds(a:float, T:float, b:float) -> float:
    # P(max_{0<=t<=T} B(t)>a | B(T)=b) using reflection principle
    if b >= a:
        return 1.0
    return np.exp(-2*a*(a-b)/T)

##################################################################################################
# Visualization functions
##################################################################################################
def plot_brownian_paths(
    times: np.ndarray,
    paths: np.ndarray,
    title: str = "Brownian Motion Paths",
    figsize: Tuple[int, int]=(12,5)
) -> plt.Figure :
    # plot multiple brownian motion paths
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(paths.shape[0]):
        ax.plot(times, paths[i], alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('B(t)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_bridge_paths(
    times: np.ndarray,
    paths: np.ndarray,
    a: float,
    b: float,
    figsize: Tuple[int, int]=(12,5)
) -> plt.Figure :
    # plot multiple brownian motion paths
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(paths.shape[0]):
        ax.plot(times, paths[i], alpha=0.7, linewidth=0.8)
    ax.scatter([times[0], times[-1]], [a,b], color='red', s=100, zorder=5, label='Fixed endpoints')
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('B(t)', fontsize=12)
    ax.set_title(f'Brownian Bridge: B(0)= {a} -> B(T)={b}', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

