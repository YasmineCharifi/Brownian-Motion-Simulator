"""
This module provides functions for statistical diagnostic including:
confidence intervals, hypothesis testing helpers, and distribution analysis 
"""

import numpy as np 
from scipy import stats
from typing import Union, Tuple, Dict, Optional, Callable
import warnings

# helper functions
def _bca_interval(data:np.ndarray, bootstrap_stats:np.ndarray, confidence_level: float )-> Tuple[float, float]:
    """
    Computing bias-corrected and accelerated bootstrap interval
    """
    n =len(data)
    origin_stat = np.mean(data)
    alpha = 1 - confidence_level
    # bias correction factor
    z0 = stats.norm.ppf(np.mean(bootstrap_stats<origin_stat))
    # the acceleration factor (jackknife estimation)
    jackknifre_stats = np.zeros(n)
    for i in range(n):
        jackknife_sample = np.delete(data,i)
        jackknifre_stats[i] = np.mean(jackknife_sample)
    jackknife_mean = np.mean(jackknifre_stats)
    num = np.sum((jackknife_mean-jackknifre_stats)**3)
    denom = 6*(np.sum((jackknife_mean-jackknifre_stats)**2)**1.5)
    if denom==0:
        acceleration = 0
    else:
        acceleration = num/denom
    # adjusting percentiles
    z_alpha_lower = stats.norm.ppf(alpha/2)
    z_alpha_upper = stats.norm.ppf(1-alpha/2)
    # another helper function to adjust percentiles
    def adjusted_percentile(z_alpha):
        numerator = z0 + z_alpha
        denominator = 1 - acceleration * numerator

        if denominator == 0:
            return 0.5

        adjusted_z = z0 + numerator / denominator
        return stats.norm.cdf(adjusted_z)
    lower_pct = adjusted_percentile(z_alpha_lower)
    upper_pct = adjusted_percentile(z_alpha_upper)

    lower_pct = np.clip(lower_pct, 0.001, 0.999)
    upper_pct = np.clip(upper_pct, 0.001, 0.999)

    ci_lower = np.percentile(bootstrap_stats,100*lower_pct)
    ci_upper = np.percentile(bootstrap_stats,100*upper_pct)

    return ci_lower, ci_upper
        

def confidence_interval_mean(data: Union[list, np.ndarray],
                              confidence_level: float=0.95, 
                              method: str='auto', 
                              n_bootstrap: int = 10000, 
                              population_std: Optional[float]=None
                              ) -> Dict[str,Union[float, str, Tuple[float, float]]]:
    """
     calculating the confidence interval for the population mean
     this function provides multiple methods for computing confidence intervals,
     autmatically selecting the most appropriate method based on the sample size
     and distribution characteristics when method is set in auto
     the methods handled are: 'auto', 't': student's t-interval (assumes approximate normally)
     'z': Z-interval (requires known population_std), 'bootstrap': bootstrap percentile method(distribution-free)
     'bootstrap_bca': bias-corrected accelerated bootstrap bootstrap
     
      the result would be a dictionnary containing:
      - point estimate: sample mean
      - ci_lower: lower band of confidence interval
      - ci_upper: upper band of confidence interval
      - confidence_level: the confidence level used
      - method: the method used for computation
      - standard error : estimated standard error of the mean
      - margin of error : half-width of the confidence interval
      - sample size : nbr of observation
    """
    # input validation
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)] #to remove the naN values
    if len(data)<2:
        raise ValueError("Need at least 2 observation to compute confidence interval")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)

    # Auto-selected method if needed
    if method == 'auto':
        if population_std is not None:
            method = 'z'
        elif n>=30:
            #chacking for normality using Shapiro-wilk
            if n<=5000: # as Shapiro Wilk works best with n less than 5000
                _, p_value = stats.shapiro(data)
                if p_value<0.05: # significant departure from normality
                    method = 'bootstrap'
                else:
                    method = 't' # applied for large n
            else:
                method = 't' 
        else:
            method = 'bootstrap' # works well with small samples for robustness
    # compute confidence interval using one of the methods 
    alpha = 1 - confidence_level
    if method == 'z':
        if population_std is None:
            raise ValueError("population_std required for z-interval method")
        se = population_std/np.sqrt(n)
        z_crit= stats.norm.ppf(1-alpha/2)
        margin = z_crit*se
        ci_lower = sample_mean - margin
        ci_upper = sample_mean + margin

    elif method == 't':
        se = sample_std/np.sqrt(n)
        t_crit = stats.t.ppf(1-alpha/2, df=n-1)
        margin = t_crit*se
        ci_lower = sample_mean - margin
        ci_upper = sample_mean + margin

    elif method in ['bootstrap','bca_bootstrap']:
        # generate bootstrap samples
        bootstrap_means = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means[i] = np.mean(bootstrap_sample)
        if method == 'bootstrap':
            #percentile method (Monte Carlo CI)
            ci_lower = np.percentile(bootstrap_means, 100*alpha/2)
            ci_upper = np.percentile(bootstrap_means, 100*(1-alpha/2))
        else:
            # BCA method: bias-correlated and accelrated
            ci_lower, ci_upper = _bca_interval(data, bootstrap_means, confidence_level)
        se = np.std(bootstrap_means)
        margin = (ci_upper-ci_lower)/2
    else:
        raise ValueError(f"Unknown Method:{method}. Use 'auto', 't', 'z', 'bootstrap', 'bca_bootstrap' ")
    if method not in ['bootstrap', 'bca_bootstrap']:
        pass
    return {
        'point_estimate': float(sample_mean),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence_level': confidence_level,
        'method': method,
        'standard_error': float(se) if 'se' in dir() else float(sample_std / np.sqrt(n)),
        'margin_of_error': float(margin) if 'margin' in dir() else float(ci_upper-ci_lower)/2,
        'sample_size': n
    }


def normality_test(data: Union[list,np.ndarray], alpha: float = 0.05) -> Dict[str, Union[float,bool,str]]:
    """
    in this function, we try to perform multiple normaliy tests and provide a summary
    """
    data =np.asarray(data)
    data = data[~np.isnan(data)] 
    n = len(data)
    results = {
        'sample_size':n,
        'skeweness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data))
    }

    #Shapiro-wilk test
    if n>=3 and n<=5000:
        shapiro_stats,p_value = stats.shapiro(data)
        results['shapiro_wilk']={
            'statistics': float(shapiro_stats),
            'p_value': float(p_value),
            'is_normal': p_value >= alpha
        }
    # D'Agastio Pearson test
    if n>=20:
        ap_stats, p_value = stats.normaltest(data)
        results['dagastio_pearson']={
            'statistics': float(ap_stats),
            'p_value': float(p_value),
            'is_normal': p_value >= alpha
        }

    # Overall Assessment
    normal_votes = sum([
        results.get('shapiro_wilk', {}).get('is_normal',True),
        results.get('dagastio_pearson', {}).get('is_normal',True),
        abs(results['skeweness'])<1,
        abs(results['kurtosis'])<2
    ])

    results['likely_normal'] = normal_votes>=3
    results['recommendation'] = 't-interval' if results['likely_normal'] else 'bootstrap'

    return results


# convenience func for quick CI computation
def quick_ci(data:Union[list, np.ndarray], confidence_level:float = 0.95)-> Tuple[float, float, float]:
    # returns from the confidence interval function just: (lower, point_estimate, upper) for quick use
    result = confidence_interval_mean(data, confidence_level= confidence_level)
    return result['ci_lower'], result['point_estimate'], result['ci_upper']

if __name__=="__main__":
    #demo
    print("="*60)
    print("CONFIDENCE INTERVAL DEMO")
    print("="*60)
 
    np.random.seed(42)
    # 1st example: normal data
    normal_data = np.random.normal(100,15, size=50)
    print("\n1. Normal Data(n=50 and the true mean=100)")
    result= confidence_interval_mean(normal_data)
    print(f"Method:{result['method']}")
    print(f"Point Estimate:{result['point_estimate']:.2f}")
    print(f" 95% CI:{result['ci_lower']:.2f}, {result['ci_upper']:.2f}")

    # 2nd example: Skewed data
    skewed_data = np.random.exponential(scale=10, size=30)
    print("\n2. Skewed Data(Exponential, n=30)")
    result= confidence_interval_mean(skewed_data)
    print(f"Method:{result['method']}")
    print(f"Point Estimate:{result['point_estimate']:.2f}")
    print(f" 95% CI:{result['ci_lower']:.2f}, {result['ci_upper']:.2f}")

    # Normality test

    print("\n3. Normality test results")
    norm_results= normality_test(skewed_data)
    print(f"Skewness:{norm_results['skeweness']:.3f}")
    print(f"Kurtosis:{norm_results['kurtosis']:.3f}")
    print(f"Likely Normal:{norm_results['likely_normal']}")
    print(f"Recommendation:{norm_results['recommendation']}")


