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

def plot_conditional_distribution(
        t: float,
        s: float,
        x_s: float,
        figsize: Tuple[int, int] = (10,5)
) -> plt.Figure:
    mean, var = conditional_distribution_future(s, t, x_s)
    std = np.sqrt(var)

    x_grid = np.linspace(mean-4*std, mean+4*std, 500)
    pdf = stats.norm.pdf(x_grid, mean, std)

    fig, ax = plt.subplots(figsize= figsize)
    ax.plot(x_grid, pdf, 'b-', linewidth=2)
    ax.fill_between(x_grid, pdf, alpha=0.3)
    ax.axvline(mean, color='r', linestyle='--', label=f'Mean = {mean}')
    ax.set_xlabel(f'B({t})', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Conditional Distribution: B({t}) | B({s}) = {x_s}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig 

def plot_bridge_envelope(
        T: float,
        a: float,
        b: float,
        n_points: int=100,
        n_std: float=2.0,
        figsize: Tuple[int, int] =(12,5)
) -> plt.Figure:
    t_values = np.linspace(0.01, T-0.01, n_points)

    means = np.array([brownian_bridge_distribution(t,T,a,b)[0] for t in t_values])
    stds = np.array([np.sqrt(brownian_bridge_distribution(t,T,a,b)[1]) for t in t_values])

    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(t_values, means, 'b-', linewidth=2, label='Mean')
    ax.fill_between(t_values, means-n_std*stds, means+n_std*stds, alpha=0.3, label=f'+/- {n_std} std')
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel(f'B(t)|B(0)={a}, B({T})={b}', fontsize=12)
    ax.set_title('Brownian Bridge: Conditional Mean +/- Std dev', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_interpolation_uncertainty(
        s: float,
        t: float,
        x_s: float,
        x_t: float,
        n_points: int=100,
        n_std: float=2.0,
        figsize: Tuple[int, int]= (12,5)
) -> plt.Figure:
    # visualizing the uncertainty between two known points
    u_values = np.linspace(s+0.01, t-0.01, n_points)

    means, stds = [], []
    for u in u_values:
        m,v = conditional_interpolation(s,t,u,x_s,x_t)
        means.append(m)
        stds.append(np.sqrt(v))
    
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(u_values, means, 'b-', linewidth=2, label='Conditional Mean')
    ax.fill_between(u_values, means-n_std*stds, means+n_std*stds, alpha=0.3, label=f'(+/-) {n_std} std')
    ax.scatter([s,t], [x_s,x_t], color='red', s=100, zorder=5, label='Known Points')
    ax.set_xlabel('Time u', fontsize=12)
    ax.set_ylabel('B(u)', fontsize=12)
    ax.set_title(f'Conditional Interpolation: B(u)|B({s})={x_s}, B({t})={x_t}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(s-0.05, t+0.05)

    plt.tight_layout()
    return fig

##################################################################################################
# Verification function : it returns theoritical and simulated statistics
##################################################################################################
def verify_bridge_distribution(
        T: float =1.0,
        a: float=0.0,
        b: float=0.0,
        t_check: float=0.5,
        n_paths: int=10000,
        n_steps: int=500,
        seed: int=42
) -> dict:
    mean_theory, var_theory = brownian_bridge_distribution(t_check,T,a,b)

    _,bridges = simulate_brownian_bridge(T,n_steps,a,b,n_paths,seed)
    step_idx = int(t_check/T*n_steps)
    samples = bridges[:, step_idx]

    return{
        'theoretical_mean': mean_theory,
        'theoretical_var': var_theory,
        'simulated_mean': np.mean(samples),
        'simulated_var': np.var(samples),
        'samples': samples
    }

##################################################################################################
# DEMO
##################################################################################################
def main():
    np.random.seed(42)

    # 1. Standard Brownian Motion
    print("="*60)
    print("1. Standard Brwonian Motion")
    print("="*60)

    times, paths = simulate_brownian_motion(T=1.0, n_steps=500, n_paths=10)
    fig = plot_brownian_paths(times, paths)
    plt.show()

    # 2. Conditional Distribution
    print("="*60)
    print("2. Conditional Distribution")
    print("="*60)

    s, x_s, t = 0.3, 1.5, 0.8
    mean, var = conditional_distribution_future(s,t,x_s)
    print(f"B({t})|B({s}) = {x_s}")
    print(f"Mean:{mean}")
    print(f"Variance:{var}")
    print(f"Std dev:{np.sqrt(var):.4f}")

    fig = plot_conditional_distribution(t,s,x_s)
    plt.show()

    # 3. Brownian Bridge
    print("="*60)
    print("3. Brownian Bridge")
    print("="*60)

    times, bridges = simulate_brownian_bridge(T=1.0, n_steps=500, a=0, b=0, n_paths=20)
    fig = plot_bridge_paths(times, bridges, a=0, b=0)
    plt.show()

    fig = plot_bridge_envelope(T=1.0, a=0,b=0)
    plt.show()

    # 4. Verify Bridge Distribution
    print("="*60)
    print("4. Verify Bridge Distribution")
    print("="*60)

    results = verify_bridge_distribution()
    print(f"Distribution of B^br(0.5) for bridge from 0 to 0:")
    print(f"theoretical: mean={results['theoretical_mean']:.4f},"
          f"variance={results['theoretical_var']:.4f}")
    print(f"simulated: mean={results['simulated_mean']:.4f},"
          f"variance={results['simulated_var']:.4f}")

    # 5. Path Interpolation
    print("="*60)
    print("5. Path Interpolation")
    print("="*60)

    s,t = 0.2, 0.8
    x_s, x_t = 0.5, 1.2
    u = 0.5

    mean, var = conditional_interpolation(s,t,u,x_s,x_t)
    print(f"Given B({s})={x_s} and B({t})={x_t}")
    print(f"Distribution of B({u}):")
    print(f"Mean: {mean:.4f}")
    print(f"Std: {np.sqrt(var):.4f}")
    
    fig = plot_interpolation_uncertainty(s,t,x_s,x_t)
    plt.show()

    # 6. Maximum Distribution
    print("="*60)
    print("6. Maximum Distribution (reflection principle)")
    print("="*60)

    a,T,b = 2.0,1.0,0.5
    prob = prob_max_exceeds(a,T,b)
    print(f"P(max B(t)>{a} |B({T})= {b}) = {prob:.6f}")

    print("\n"+"="*60)
    print("Summary")
    print("="*60)

if __name__=="__main__":
    main()
