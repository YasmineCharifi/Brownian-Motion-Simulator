"""Reusable utility functions for probability distributions and simulations and statistical analysis."""

from collections import Counter
from typing import List, Tuple, Dict, Union
import math

def empirical_mean(data: List[Union[int, float]]) -> float:
    """Calculate the empirical mean of a dataset."""
    if not data:
        raise ValueError("Data list is empty.")
    return sum(data) / len(data)

def empirical_variance(data: List[Union[int, float]], ddof: int = 1) -> float:
    """Calculate the empirical variance of a dataset."""
    n = len(data)
    if n == 0:
        raise ValueError("Data list is empty.")
    if n <= ddof:
        raise ValueError("Degrees of freedom must be less than the number of data points.")
    mean = empirical_mean(data)
    squared_diffs = [(x - mean) ** 2 for x in data]
    return sum(squared_diffs) / (n - ddof)

def empirical_std_dev(data: List[Union[int, float]], ddof: int = 1) -> float:
    """Calculate the empirical standard deviation of a dataset."""
    variance = empirical_variance(data, ddof)
    return math.sqrt(variance)

def empirical_covariance(data_x: List[Union[int, float]], data_y: List[Union[int, float]], ddof: int = 1) -> float:
    """Calculate the empirical covariance between two datasets."""
    if len(data_x) != len(data_y):
        raise ValueError("Data lists must have the same length.")
    n = len(data_x)
    if n == 0:
        raise ValueError("Data lists are empty.")
    if n <= ddof:
        raise ValueError("Degrees of freedom must be less than the number of data points.")
    mean_x = empirical_mean(data_x)
    mean_y = empirical_mean(data_y)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(data_x, data_y)) / (n - ddof)
    return covariance

def empirical_correlation(data_x: List[Union[int, float]], data_y: List[Union[int, float]], ddof: int = 1) -> float:
    """Calculate the empirical correlation coefficient between two datasets."""
    covariance = empirical_covariance(data_x, data_y, ddof)
    std_dev_x = empirical_std_dev(data_x, ddof)
    std_dev_y = empirical_std_dev(data_y, ddof)
    if std_dev_x == 0 or std_dev_y == 0:
        raise ValueError("Standard deviation of one or both datasets is zero.")
    return covariance / (std_dev_x * std_dev_y)

def compute_pmf(data: List[Union[int, float]]) -> Dict[Union[int, float], float]:
    """Compute the probability mass function (PMF) of a discrete dataset."""
    if not data:
        raise ValueError("Data list is empty.")
    count = Counter(data)
    total = len(data)
    pmf = {k: v / total for k, v in count.items()}
    return pmf

def compute_cdf(data: List[Union[int, float]]) -> List[Tuple[Union[int, float], float]]:
    """Compute the cumulative distribution function (CDF) of a dataset."""
    if not data:
        raise ValueError("Data list is empty.")
    pmf = compute_pmf(data)
    sorted_data = sorted(pmf.keys())
    cdf = []
    cumulative_count = 0.0
    for value in sorted_data:
        cumulative_count += pmf[value]
        cdf.append((value, cumulative_count))
    return cdf

def expected_value(pmf: Dict[Union[int, float], float]) -> float:
    """Calculate the expected value of a dataset."""
    return sum(value * prob for value, prob in pmf.items())

def variance_from_pmf(pmf: Dict[Union[int, float], float]) -> float:
    """Calculate the variance of a dataset from its PMF."""
    mean = expected_value(pmf)
    return sum(prob * (value - mean) ** 2 for value, prob in pmf.items())

# function for running quick simulations of discrete distributions
def simulate_discrete_distribution(data: List[Union[int, float]], name: str) -> Dict[str, Union[str, int]]:
    """Simulate samples from a discrete distribution defined by the input data."""
    return {
        'name': name,
        'count': len(data),
        'mean': empirical_mean(data),
        'variance': empirical_variance(data),
        'std_dev': empirical_std_dev(data),
        'min': min(data),
        'max': max(data),
    }

if __name__ == "__main__":
    # Example usage
    data = [1, 2, 2, 3, 3, 3, 4]
    print("Empirical Mean:", empirical_mean(data))
    print("Empirical Variance:", empirical_variance(data))
    print("Empirical Standard Deviation:", empirical_std_dev(data))
    print("PMF:", compute_pmf(data))
    print("CDF:", compute_cdf(data))
    print("Simulated Distribution:", simulate_discrete_distribution(data, "Example"))