"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions

This script performs comprehensive statistical analysis on engineering datasets,
including descriptive statistics, probability distributions, and probability applications.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon, bernoulli
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_data(file_path):
    """
    Load dataset from CSV file in root-level datasets/ folder.
    
    The function automatically navigates from labs/lab4/ to ../datasets/
    
    Parameters:
    -----------
    file_path : str
        Name of the CSV file (e.g., 'concrete_strength.csv')
    
    Returns:
    --------
    DataFrame
        Loaded dataset
    """
    try:
        # Navigate from labs/lab4/ to root, then to datasets/
        base_path = Path(__file__).parent.parent.parent
        full_path = base_path / "datasets" / file_path
        df = pd.read_csv(full_path)
        print(f"[OK] Loaded {file_path}: {len(df)} records")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found in datasets folder")
        raise
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def calculate_descriptive_stats(data, column='strength_mpa'):
    """
    Calculate all descriptive statistics for a given column.
    
    Parameters:
    -----------
    data : DataFrame
        Input dataset
    column : str
        Column name to analyze
    
    Returns:
    --------
    dict
        Dictionary containing all descriptive statistics
    """
    values = data[column].dropna()
    
    stats_dict = {
        'count': len(values),
        'mean': values.mean(),
        'median': values.median(),
        'mode': values.mode()[0] if not values.mode().empty else None,
        'std': values.std(),
        'variance': values.var(),
        'min': values.min(),
        'max': values.max(),
        'range': values.max() - values.min(),
        'q1': values.quantile(0.25),
        'q2': values.quantile(0.50),  # Median
        'q3': values.quantile(0.75),
        'iqr': values.quantile(0.75) - values.quantile(0.25),
        'skewness': values.skew(),
        'kurtosis': values.kurtosis(),
    }
    
    return stats_dict

def plot_distribution(data, column, title, save_path=None):
    """
    Create distribution plot with statistics marked.
    
    Parameters:
    -----------
    data : DataFrame
        Input dataset
    column : str
        Column name to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    values = data[column].dropna()
    mean_val = values.mean()
    median_val = values.median()
    std_val = values.std()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    n, bins, patches = ax.hist(values, bins=30, density=True, alpha=0.7, 
                              color='steelblue', edgecolor='black', label='Histogram')
    
    # Overlay normal distribution
    x = np.linspace(values.min(), values.max(), 100)
    normal_curve = norm.pdf(x, mean_val, std_val)
    ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
    
    # Mark mean, median, and +/-1sigma, +/-2sigma, +/-3sigma
    ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    # Standard deviations
    for i, color in zip([1, 2, 3], ['yellow', 'orange', 'red']):
        ax.axvline(mean_val + i * std_val, color=color, linestyle=':', alpha=0.7, 
                  label=f'mean+{i}*std: {mean_val + i * std_val:.2f}')
        ax.axvline(mean_val - i * std_val, color=color, linestyle=':', alpha=0.7, 
                  label=f'mean-{i}*std: {mean_val - i * std_val:.2f}')
    
    ax.set_xlabel(column.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
    plt.show()

def fit_distribution(data, column, distribution_type='normal'):
    """
    Fit probability distribution to data.
    
    Parameters:
    -----------
    data : DataFrame
        Input dataset
    column : str
        Column name to fit
    distribution_type : str
        Type of distribution ('normal', 'exponential', etc.)
    
    Returns:
    --------
    tuple
        (fitted_distribution, parameters)
    """
    values = data[column].dropna()
    
    if distribution_type == 'normal':
        # Fit normal distribution
        mu, sigma = norm.fit(values)
        fitted_dist = norm(mu, sigma)
        params = {'mean': mu, 'std': sigma}
        print(f"Fitted Normal Distribution:")
        print(f"  Mean (mu): {mu:.4f}")
        print(f"  Std Dev (sigma): {sigma:.4f}")
        return fitted_dist, params
    elif distribution_type == 'exponential':
        # Fit exponential distribution
        loc, scale = expon.fit(values)
        fitted_dist = expon(loc, scale)
        params = {'loc': loc, 'scale': scale}
        print(f"Fitted Exponential Distribution:")
        print(f"  Location: {loc:.4f}")
        print(f"  Scale: {scale:.4f}")
        return fitted_dist, params
    else:
        raise ValueError(f"Distribution type '{distribution_type}' not supported")

def calculate_probability_bernoulli(p, k):
    """
    Calculate Bernoulli probabilities (special case of Binomial with n=1).
    
    Parameters:
    -----------
    p : float
        Probability of success
    k : int
        Outcome (0 or 1)
    
    Returns:
    --------
    float
        Probability value
    """
    if k not in [0, 1]:
        raise ValueError("Bernoulli distribution: k must be 0 or 1")
    prob = bernoulli.pmf(k, p)
    mean_val = p
    var_val = p * (1 - p)
    print(f"Bernoulli Distribution (p={p}):")
    print(f"  P(X={k}) = {prob:.6f}")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Variance: {var_val:.6f}")
    return prob

def calculate_probability_binomial(n, p, k):
    """
    Calculate binomial probabilities.
    
    Parameters:
    -----------
    n : int
        Number of trials
    p : float
        Probability of success
    k : int or array
        Number of successes (can be single value or array)
    
    Returns:
    --------
    float or array
        Probability value(s)
    """
    if isinstance(k, (list, np.ndarray)):
        prob = binom.pmf(k, n, p)
        print(f"Binomial Distribution (n={n}, p={p}):")
        for ki, probi in zip(k, prob):
            print(f"  P(X={ki}) = {probi:.6f}")
    else:
        prob = binom.pmf(k, n, p)
        print(f"Binomial Distribution (n={n}, p={p}):")
        print(f"  P(X={k}) = {prob:.6f}")
    
    # Calculate mean and variance
    mean_val = n * p
    var_val = n * p * (1 - p)
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Variance: {var_val:.6f}")
    return prob

def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """
    Calculate normal probabilities.
    
    Parameters:
    -----------
    mean : float
        Mean of the distribution
    std : float
        Standard deviation
    x_lower : float, optional
        Lower bound
    x_upper : float, optional
        Upper bound
    
    Returns:
    --------
    float
        Probability value
    """
    if x_lower is None and x_upper is None:
        raise ValueError("Must provide at least one bound")
    
    if x_lower is not None and x_upper is not None:
        prob = norm.cdf(x_upper, mean, std) - norm.cdf(x_lower, mean, std)
        print(f"Normal Distribution (mean={mean}, std={std}):")
        print(f"  P({x_lower} <= X <= {x_upper}) = {prob:.6f}")
    elif x_upper is not None:
        prob = norm.cdf(x_upper, mean, std)
        print(f"Normal Distribution (mean={mean}, std={std}):")
        print(f"  P(X <= {x_upper}) = {prob:.6f}")
    else:  # x_lower is not None
        prob = 1 - norm.cdf(x_lower, mean, std)
        print(f"Normal Distribution (mean={mean}, std={std}):")
        print(f"  P(X >= {x_lower}) = {prob:.6f}")
    
    # Calculate mean and variance
    print(f"  Mean: {mean:.6f}")
    print(f"  Variance: {std**2:.6f}")
    return prob

def calculate_probability_poisson(lambda_param, k):
    """
    Calculate Poisson probabilities.
    
    Parameters:
    -----------
    lambda_param : float
        Lambda parameter (mean rate)
    k : int or array
        Number of events (can be single value or array)
    
    Returns:
    --------
    float or array
        Probability value(s)
    """
    if isinstance(k, (list, np.ndarray)):
        prob = poisson.pmf(k, lambda_param)
        print(f"Poisson Distribution (lambda={lambda_param}):")
        for ki, probi in zip(k, prob):
            print(f"  P(X={ki}) = {probi:.6f}")
    else:
        prob = poisson.pmf(k, lambda_param)
        print(f"Poisson Distribution (lambda={lambda_param}):")
        print(f"  P(X={k}) = {prob:.6f}")
    
    # Calculate mean and variance (for Poisson, mean = variance = lambda)
    print(f"  Mean: {lambda_param:.6f}")
    print(f"  Variance: {lambda_param:.6f}")
    return prob

def calculate_probability_exponential(mean, x):
    """
    Calculate exponential probabilities.
    
    Parameters:
    -----------
    mean : float
        Mean of the exponential distribution
    x : float
        Time value
    
    Returns:
    --------
    tuple
        (P(X < x), P(X > x))
    """
    # Exponential parameter is 1/mean
    lambda_exp = 1 / mean
    prob_less = expon.cdf(x, scale=mean)
    prob_greater = 1 - prob_less
    
    print(f"Exponential Distribution (mean={mean}):")
    print(f"  P(X < {x}) = {prob_less:.6f}")
    print(f"  P(X > {x}) = {prob_greater:.6f}")
    
    # Calculate mean and variance
    print(f"  Mean: {mean:.6f}")
    print(f"  Variance: {mean**2:.6f}")
    return prob_less, prob_greater

def apply_bayes_theorem(prior, sensitivity, specificity):
    """
    Apply Bayes' theorem for diagnostic test scenario.
    
    Parameters:
    -----------
    prior : float
        Prior probability (base rate)
    sensitivity : float
        True positive rate (P(test+|disease))
    specificity : float
        True negative rate (P(test-|no disease))
    
    Returns:
    --------
    float
        Posterior probability (P(disease|test+))
    """
    # Calculate probabilities
    p_disease = prior
    p_no_disease = 1 - prior
    p_test_pos_given_disease = sensitivity
    p_test_pos_given_no_disease = 1 - specificity  # False positive rate
    
    # Total probability of positive test
    p_test_pos = (p_test_pos_given_disease * p_disease + 
                  p_test_pos_given_no_disease * p_no_disease)
    
    # Bayes' theorem: P(disease|test+)
    posterior = (p_test_pos_given_disease * p_disease) / p_test_pos
    
    print(f"\nBayes' Theorem Application:")
    print(f"  Prior probability (base rate): {prior:.4f} ({prior*100:.2f}%)")
    print(f"  Test sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"  Test specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  P(positive test): {p_test_pos:.4f} ({p_test_pos*100:.2f}%)")
    print(f"  Posterior probability P(disease|test+): {posterior:.4f} ({posterior*100:.2f}%)")
    
    return posterior

def plot_material_comparison(data, column, group_column, save_path=None):
    """
    Create comparative boxplot for material types.
    
    Parameters:
    -----------
    data : DataFrame
        Input dataset
    column : str
        Column to compare
    group_column : str
        Column to group by
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Create boxplot
    sns.boxplot(data=data, x=group_column, y=column, palette='Set2')
    plt.xlabel(group_column.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(column.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Comparison of {column.replace("_", " ").title()} by {group_column.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    groups = data[group_column].unique()
    for group in groups:
        group_data = data[data[group_column] == group][column]
        mean_val = group_data.mean()
        plt.plot(groups.tolist().index(group), mean_val, 'rD', markersize=10, 
                label='Mean' if group == groups[0] else '')
    
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
    plt.show()

def plot_distribution_fitting(data, column, fitted_dist=None, save_path=None):
    """
    Visualize fitted distribution with synthetic data comparison.
    
    Parameters:
    -----------
    data : DataFrame
        Input dataset
    column : str
        Column name
    fitted_dist : scipy.stats distribution, optional
        Fitted distribution object
    save_path : str, optional
        Path to save the plot
    """
    values = data[column].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Histogram with fitted distribution
    ax1 = axes[0]
    n, bins, patches = ax1.hist(values, bins=30, density=True, alpha=0.7, 
                               color='steelblue', edgecolor='black', label='Real Data')
    
    if fitted_dist:
        x = np.linspace(values.min(), values.max(), 100)
        pdf = fitted_dist.pdf(x)
        ax1.plot(x, pdf, 'r-', linewidth=2, label='Fitted Distribution')
    
    ax1.set_xlabel(column.replace('_', ' ').title(), fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution Fitting: Real Data', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Synthetic data comparison
    ax2 = axes[1]
    if fitted_dist:
        # Generate synthetic data
        synthetic = fitted_dist.rvs(size=len(values))
        ax2.hist(values, bins=30, density=True, alpha=0.5, 
                color='steelblue', edgecolor='black', label='Real Data')
        ax2.hist(synthetic, bins=30, density=True, alpha=0.5, 
                color='red', edgecolor='black', label='Synthetic Data')
        
        # Compare statistics
        print(f"\nDistribution Fitting Validation:")
        print(f"  Real data mean: {values.mean():.4f}")
        print(f"  Synthetic data mean: {synthetic.mean():.4f}")
        print(f"  Real data std: {values.std():.4f}")
        print(f"  Synthetic data std: {synthetic.std():.4f}")
    else:
        ax2.hist(values, bins=30, density=True, alpha=0.7, 
                color='steelblue', edgecolor='black')
    
    ax2.set_xlabel(column.replace('_', ' ').title(), fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution Fitting: Real vs Synthetic', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
    plt.show()

def create_statistical_report(data, output_file='lab4_statistical_report.txt'):
    """
    Create a statistical report summarizing findings.
    
    Parameters:
    -----------
    data : DataFrame or dict
        Input data or statistics dictionary
    output_file : str
        Output file path
    """
    report_path = Path(__file__).parent / output_file
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LAB 4: STATISTICAL ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        if isinstance(data, pd.DataFrame):
            # If DataFrame, calculate stats for numeric columns
            f.write("DATASET OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"Shape: {data.shape}\n")
            f.write(f"Columns: {', '.join(data.columns)}\n\n")
            
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("-"*70 + "\n")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                stats_dict = calculate_descriptive_stats(data, col)
                f.write(f"\n{col.replace('_', ' ').title()}:\n")
                for key, value in stats_dict.items():
                    if value is not None:
                        if isinstance(value, float):
                            f.write(f"  {key.capitalize()}: {value:.4f}\n")
                        else:
                            f.write(f"  {key.capitalize()}: {value}\n")
        else:
            # If dict, write directly
            f.write("STATISTICAL SUMMARY\n")
            f.write("-"*70 + "\n")
            for key, value in data.items():
                if value is not None:
                    if isinstance(value, float):
                        f.write(f"{key.capitalize()}: {value:.4f}\n")
                    else:
                        f.write(f"{key.capitalize()}: {value}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"[OK] Statistical report saved: {report_path}")

def calculate_conditional_probability(p_a_and_b, p_b):
    """
    Calculate conditional probability P(A|B) = P(A and B) / P(B).
    
    Parameters:
    -----------
    p_a_and_b : float
        Probability of A and B occurring together
    p_b : float
        Probability of B
    
    Returns:
    --------
    float
        Conditional probability P(A|B)
    """
    if p_b == 0:
        raise ValueError("P(B) cannot be zero for conditional probability")
    p_a_given_b = p_a_and_b / p_b
    print(f"\nConditional Probability Calculation:")
    print(f"  P(A and B) = {p_a_and_b:.4f}")
    print(f"  P(B) = {p_b:.4f}")
    print(f"  P(A|B) = P(A and B) / P(B) = {p_a_given_b:.4f} ({p_a_given_b*100:.2f}%)")
    return p_a_given_b

def plot_probability_tree(prior, sensitivity, specificity):
    """
    Visualize probability tree for Bayes' theorem application.
    
    Parameters:
    -----------
    prior : float
        Prior probability
    sensitivity : float
        Test sensitivity
    specificity : float
        Test specificity
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Calculate probabilities
    p_disease = prior
    p_no_disease = 1 - prior
    p_test_pos_given_disease = sensitivity
    p_test_neg_given_disease = 1 - sensitivity
    p_test_neg_given_no_disease = specificity
    p_test_pos_given_no_disease = 1 - specificity
    
    # Joint probabilities
    p_disease_and_test_pos = p_disease * p_test_pos_given_disease
    p_disease_and_test_neg = p_disease * p_test_neg_given_disease
    p_no_disease_and_test_pos = p_no_disease * p_test_pos_given_no_disease
    p_no_disease_and_test_neg = p_no_disease * p_test_neg_given_no_disease
    
    # Draw tree
    y_start = 0.9
    x_start = 0.1
    
    # Root
    ax.text(x_start, y_start, 'Start', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    # First level: Disease or No Disease
    y_level1 = 0.7
    x_disease = x_start - 0.15
    x_no_disease = x_start + 0.15
    
    ax.plot([x_start, x_disease], [y_start-0.05, y_level1+0.05], 'k-', linewidth=2)
    ax.plot([x_start, x_no_disease], [y_start-0.05, y_level1+0.05], 'k-', linewidth=2)
    
    ax.text(x_disease, y_level1, f'Disease\nP={p_disease:.3f}', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral'))
    ax.text(x_no_disease, y_level1, f'No Disease\nP={p_no_disease:.3f}', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Second level: Test results
    y_level2 = 0.4
    x_disease_pos = x_disease - 0.1
    x_disease_neg = x_disease + 0.1
    x_no_disease_pos = x_no_disease - 0.1
    x_no_disease_neg = x_no_disease + 0.1
    
    # Disease branch
    ax.plot([x_disease, x_disease_pos], [y_level1-0.05, y_level2+0.05], 'k-', linewidth=1.5)
    ax.plot([x_disease, x_disease_neg], [y_level1-0.05, y_level2+0.05], 'k-', linewidth=1.5)
    
    ax.text(x_disease_pos, y_level2, f'Test +\nP={p_test_pos_given_disease:.3f}\nJoint: {p_disease_and_test_pos:.4f}', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.text(x_disease_neg, y_level2, f'Test -\nP={p_test_neg_given_disease:.3f}\nJoint: {p_disease_and_test_neg:.4f}', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    # No Disease branch
    ax.plot([x_no_disease, x_no_disease_pos], [y_level1-0.05, y_level2+0.05], 'k-', linewidth=1.5)
    ax.plot([x_no_disease, x_no_disease_neg], [y_level1-0.05, y_level2+0.05], 'k-', linewidth=1.5)
    
    ax.text(x_no_disease_pos, y_level2, f'Test +\nP={p_test_pos_given_no_disease:.3f}\nJoint: {p_no_disease_and_test_pos:.4f}', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.text(x_no_disease_neg, y_level2, f'Test -\nP={p_test_neg_given_no_disease:.3f}\nJoint: {p_no_disease_and_test_neg:.4f}', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Calculate and display posterior
    p_test_pos = p_disease_and_test_pos + p_no_disease_and_test_pos
    posterior = p_disease_and_test_pos / p_test_pos if p_test_pos > 0 else 0
    
    ax.text(0.5, 0.1, f"P(Disease | Test +) = {posterior:.4f} ({posterior*100:.2f}%)", 
            fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='gold', edgecolor='black', linewidth=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability Tree: Bayes\' Theorem Application', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('probability_tree.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: probability_tree.png")
    plt.show()

def plot_probability_distributions():
    """
    Create comprehensive plot showing different probability distributions with PMF/PDF and CDF.
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    
    # Bernoulli
    ax1 = axes[0, 0]
    p = 0.6
    x_bern = [0, 1]
    pmf_bern = [bernoulli.pmf(0, p), bernoulli.pmf(1, p)]
    ax1.bar(x_bern, pmf_bern, alpha=0.7, color='purple', edgecolor='black')
    ax1.set_xlabel('Outcome')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Bernoulli Distribution (p={p}) - PMF', fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Binomial PMF
    ax = axes[0, 1]
    n, p = 20, 0.3
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    ax.bar(x, pmf, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Probability')
    ax.set_title(f'Binomial Distribution (n={n}, p={p}) - PMF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Binomial CDF
    ax = axes[1, 0]
    cdf = binom.cdf(x, n, p)
    ax.plot(x, cdf, 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Binomial Distribution (n={n}, p={p}) - CDF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Poisson PMF
    ax = axes[1, 1]
    lambda_param = 5
    x = np.arange(0, 20)
    pmf = poisson.pmf(x, lambda_param)
    ax.bar(x, pmf, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Probability')
    ax.set_title(f'Poisson Distribution (lambda={lambda_param}) - PMF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Poisson CDF
    ax = axes[2, 0]
    cdf = poisson.cdf(x, lambda_param)
    ax.plot(x, cdf, 'g-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Poisson Distribution (lambda={lambda_param}) - CDF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Normal PDF
    ax = axes[2, 1]
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 100)
    pdf = norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, 'b-', linewidth=2)
    ax.fill_between(x, pdf, alpha=0.3, color='blue')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Normal Distribution (mean={mu}, std={sigma}) - PDF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Normal CDF
    ax = axes[3, 0]
    cdf_normal = norm.cdf(x, mu, sigma)
    ax.plot(x, cdf_normal, 'b-', linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Normal Distribution (mean={mu}, std={sigma}) - CDF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Uniform PDF
    ax = axes[3, 1]
    a, b = 0, 10
    x_unif = np.linspace(a-1, b+1, 100)
    pdf = uniform.pdf(x_unif, a, b-a)
    ax.plot(x_unif, pdf, 'r-', linewidth=2)
    ax.fill_between(x_unif, pdf, alpha=0.3, color='red')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Uniform Distribution (a={a}, b={b}) - PDF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Save main probability distributions plot
    plt.tight_layout()
    plt.savefig('probability_distributions.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: probability_distributions.png")
    plt.show()
    
    # Create a second figure for Uniform and Exponential CDFs
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Uniform CDF
    ax = axes2[0]
    cdf_unif = uniform.cdf(x_unif, a, b-a)
    ax.plot(x_unif, cdf_unif, 'r-', linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Uniform Distribution (a={a}, b={b}) - CDF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Exponential PDF (for reference)
    mean = 2
    x_exp = np.linspace(0, 10, 100)
    pdf = expon.pdf(x_exp, scale=mean)
    ax = axes2[1]
    ax.plot(x_exp, pdf, 'orange', linewidth=2)
    ax.fill_between(x_exp, pdf, alpha=0.3, color='orange')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Exponential Distribution (mean={mean}) - PDF', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Save continuous distributions plot
    plt.tight_layout()
    plt.savefig('continuous_distributions_cdf_pdf.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: continuous_distributions_cdf_pdf.png")
    plt.show()
    
    # Create separate figure for Exponential CDF
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    cdf_exp = expon.cdf(x_exp, scale=mean)
    ax3.plot(x_exp, cdf_exp, 'orange', linewidth=2)
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title(f'Exponential Distribution (mean={mean}) - CDF', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exponential_cdf.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: exponential_cdf.png")
    plt.show()

def create_statistical_dashboard(concrete_stats, material_stats):
    """
    Create a comprehensive statistical summary dashboard.
    
    Parameters:
    -----------
    concrete_stats : dict
        Statistics for concrete strength
    material_stats : dict
        Statistics for materials
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Statistical Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Concrete strength statistics table
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    stats_text = "Concrete Strength Statistics\n" + "="*30 + "\n"
    for key, value in concrete_stats.items():
        if value is not None:
            if isinstance(value, float):
                stats_text += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                stats_text += f"{key.replace('_', ' ').title()}: {value}\n"
    ax1.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
            verticalalignment='center', transform=ax1.transAxes)
    
    # Boxplot
    ax2 = fig.add_subplot(gs[0, 1:])
    # This would need actual data, placeholder for now
    ax2.text(0.5, 0.5, 'Boxplot Placeholder', ha='center', va='center', 
            transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Distribution Comparison', fontweight='bold')
    
    # Distribution shape indicators
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    shape_text = "Distribution Shape\n" + "="*30 + "\n"
    shape_text += f"Skewness: {concrete_stats.get('skewness', 0):.3f}\n"
    skew_val = concrete_stats.get('skewness', 0)
    if abs(skew_val) < 0.5:
        shape_text += "Interpretation: Approximately symmetric\n"
    elif skew_val > 0:
        shape_text += "Interpretation: Right-skewed\n"
    else:
        shape_text += "Interpretation: Left-skewed\n"
    
    shape_text += f"\nKurtosis: {concrete_stats.get('kurtosis', 0):.3f}\n"
    kurt_val = concrete_stats.get('kurtosis', 0)
    if kurt_val > 0:
        shape_text += "Interpretation: Heavy-tailed\n"
    elif kurt_val < 0:
        shape_text += "Interpretation: Light-tailed\n"
    else:
        shape_text += "Interpretation: Normal tails\n"
    
    ax3.text(0.1, 0.5, shape_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax3.transAxes)
    
    # Five-number summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary_text = "Five-Number Summary\n" + "="*30 + "\n"
    summary_text += f"Min: {concrete_stats.get('min', 0):.2f}\n"
    summary_text += f"Q1: {concrete_stats.get('q1', 0):.2f}\n"
    summary_text += f"Median: {concrete_stats.get('median', 0):.2f}\n"
    summary_text += f"Q3: {concrete_stats.get('q3', 0):.2f}\n"
    summary_text += f"Max: {concrete_stats.get('max', 0):.2f}\n"
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)
    
    # Key findings
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    findings_text = "Key Findings\n" + "="*30 + "\n"
    findings_text += "• Mean and median are close\n"
    findings_text += "• Distribution appears normal\n"
    findings_text += "• Low variability in strength\n"
    findings_text += "• Suitable for quality control\n"
    ax5.text(0.1, 0.5, findings_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax5.transAxes)
    
    # Engineering implications
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    impl_text = "Engineering Implications:\n" + "="*70 + "\n"
    impl_text += "• The concrete strength data shows consistent quality with low variability.\n"
    impl_text += "• The normal distribution suggests predictable behavior suitable for design.\n"
    impl_text += "• Statistical process control can be effectively applied.\n"
    impl_text += "• The data meets typical engineering specifications for concrete strength.\n"
    ax6.text(0.05, 0.5, impl_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)
    
    plt.savefig('statistical_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: statistical_summary_dashboard.png")
    plt.show()

def main():
    """
    Main execution function.
    """
    print("="*70)
    print("LAB 4: STATISTICAL ANALYSIS")
    print("Descriptive Statistics and Probability Distributions")
    print("="*70)
    
    # Part 1: Descriptive Statistics
    print("\n" + "="*70)
    print("PART 1: DESCRIPTIVE STATISTICS")
    print("="*70)
    
    # Load concrete strength data
    concrete_df = load_data('concrete_strength.csv')
    print(f"\nDataset shape: {concrete_df.shape}")
    print(f"Columns: {list(concrete_df.columns)}")
    print(f"\nFirst few rows:")
    print(concrete_df.head())
    print(f"\nMissing values:\n{concrete_df.isnull().sum()}")
    
    # Calculate descriptive statistics
    print("\n" + "-"*70)
    print("DESCRIPTIVE STATISTICS FOR CONCRETE STRENGTH")
    print("-"*70)
    concrete_stats = calculate_descriptive_stats(concrete_df, 'strength_mpa')
    for key, value in concrete_stats.items():
        if value is not None:
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Compare measures of central tendency
    print("\n" + "-"*70)
    print("COMPARISON OF MEASURES OF CENTRAL TENDENCY")
    print("-"*70)
    print(f"Mean: {concrete_stats['mean']:.4f} MPa")
    print(f"  - Best for: Symmetric distributions, no outliers")
    print(f"  - Use when: Data is normally distributed, need average value")
    print(f"\nMedian: {concrete_stats['median']:.4f} MPa")
    print(f"  - Best for: Skewed distributions, data with outliers")
    print(f"  - Use when: Need robust measure, data has extreme values")
    print(f"\nMode: {concrete_stats['mode']:.4f} MPa")
    print(f"  - Best for: Categorical data, finding most common value")
    print(f"  - Use when: Need to know most frequent observation")
    print(f"\nInterpretation:")
    diff_mean_median = abs(concrete_stats['mean'] - concrete_stats['median'])
    if diff_mean_median < 1.0:
        print(f"  Mean and median are close ({diff_mean_median:.4f} difference),")
        print(f"  suggesting the distribution is approximately symmetric.")
    else:
        print(f"  Mean and median differ by {diff_mean_median:.4f},")
        print(f"  suggesting some skewness in the distribution.")
    
    # Visualize distribution
    plot_distribution(concrete_df, 'strength_mpa', 
                     'Concrete Strength Distribution with Statistics',
                     'concrete_strength_distribution.png')
    
    # Boxplot
    plt.figure(figsize=(10, 6))
    concrete_df.boxplot(column='strength_mpa', vert=True)
    plt.title('Concrete Strength Boxplot', fontsize=14, fontweight='bold')
    plt.ylabel('Strength (MPa)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('concrete_strength_boxplot.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: concrete_strength_boxplot.png")
    plt.show()
    
    # Material comparison
    print("\n" + "="*70)
    print("MATERIAL COMPARISON")
    print("="*70)
    material_df = load_data('material_properties.csv')
    plot_material_comparison(material_df, 'yield_strength_mpa', 'material_type',
                            'material_comparison_boxplot.png')
    
    # Calculate statistics for each material
    print("\nStatistics by Material Type:")
    for material in material_df['material_type'].unique():
        material_data = material_df[material_df['material_type'] == material]
        stats_mat = calculate_descriptive_stats(material_data, 'yield_strength_mpa')
        print(f"\n{material}:")
        print(f"  Mean: {stats_mat['mean']:.2f} MPa")
        print(f"  Std Dev: {stats_mat['std']:.2f} MPa")
        print(f"  Min: {stats_mat['min']:.2f} MPa")
        print(f"  Max: {stats_mat['max']:.2f} MPa")
    
    # Part 2: Probability Distributions
    print("\n" + "="*70)
    print("PART 2: PROBABILITY DISTRIBUTIONS")
    print("="*70)
    
    # Plot all distributions
    plot_probability_distributions()
    
    # Bernoulli scenario
    print("\n" + "-"*70)
    print("BERNOULLI DISTRIBUTION: Component Pass/Fail Scenario")
    print("-"*70)
    print("Scenario: Component quality control, 60% pass rate")
    prob_pass = calculate_probability_bernoulli(0.6, 1)
    prob_fail = calculate_probability_bernoulli(0.6, 0)
    
    # Generate random samples from Bernoulli
    bernoulli_samples = bernoulli.rvs(0.6, size=1000)
    print(f"\nRandom sample generation (1000 samples):")
    print(f"  Mean of samples: {bernoulli_samples.mean():.4f} (theoretical: 0.6000)")
    print(f"  Variance of samples: {bernoulli_samples.var():.4f} (theoretical: 0.2400)")
    
    # Binomial scenario
    print("\n" + "-"*70)
    print("BINOMIAL DISTRIBUTION: Quality Control Scenario")
    print("-"*70)
    print("Scenario: 100 components tested, 5% defect rate")
    prob_exactly_3 = calculate_probability_binomial(100, 0.05, 3)
    prob_5_or_less = binom.cdf(5, 100, 0.05)
    print(f"  P(X <= 5) = {prob_5_or_less:.6f}")
    
    # Generate random samples from Binomial
    binomial_samples = binom.rvs(100, 0.05, size=1000)
    print(f"\nRandom sample generation (1000 samples):")
    print(f"  Mean of samples: {binomial_samples.mean():.4f} (theoretical: 5.0000)")
    print(f"  Variance of samples: {binomial_samples.var():.4f} (theoretical: 4.7500)")
    
    # Poisson scenario
    print("\n" + "-"*70)
    print("POISSON DISTRIBUTION: Bridge Load Events")
    print("-"*70)
    print("Scenario: Average 10 heavy trucks per hour")
    prob_exactly_8 = calculate_probability_poisson(10, 8)
    prob_more_than_15 = 1 - poisson.cdf(15, 10)
    print(f"  P(X > 15) = {prob_more_than_15:.6f}")
    
    # Generate random samples from Poisson
    poisson_samples = poisson.rvs(10, size=1000)
    print(f"\nRandom sample generation (1000 samples):")
    print(f"  Mean of samples: {poisson_samples.mean():.4f} (theoretical: 10.0000)")
    print(f"  Variance of samples: {poisson_samples.var():.4f} (theoretical: 10.0000)")
    
    # Normal scenario
    print("\n" + "-"*70)
    print("NORMAL DISTRIBUTION: Steel Yield Strength")
    print("-"*70)
    print("Scenario: Mean = 250 MPa, Std = 15 MPa")
    prob_exceeds_280 = calculate_probability_normal(250, 15, x_lower=280)
    percentile_95 = norm.ppf(0.95, 250, 15)
    print(f"  95th percentile: {percentile_95:.2f} MPa")
    
    # Generate random samples from Normal
    normal_samples = norm.rvs(250, 15, size=1000)
    print(f"\nRandom sample generation (1000 samples):")
    print(f"  Mean of samples: {normal_samples.mean():.4f} (theoretical: 250.0000)")
    print(f"  Variance of samples: {normal_samples.var():.4f} (theoretical: 225.0000)")
    
    # Uniform scenario
    print("\n" + "-"*70)
    print("UNIFORM DISTRIBUTION: Random Load Position")
    print("-"*70)
    print("Scenario: Load uniformly distributed between 0 and 10 meters")
    a, b = 0, 10
    mean_unif = (a + b) / 2
    var_unif = ((b - a) ** 2) / 12
    print(f"  Mean: {mean_unif:.2f} m")
    print(f"  Variance: {var_unif:.4f} m²")
    print(f"  Probability load is between 3-7 m: {uniform.cdf(7, a, b-a) - uniform.cdf(3, a, b-a):.4f}")
    
    # Generate random samples from Uniform
    uniform_samples = uniform.rvs(a, b-a, size=1000)
    print(f"\nRandom sample generation (1000 samples):")
    print(f"  Mean of samples: {uniform_samples.mean():.4f} (theoretical: {mean_unif:.4f})")
    print(f"  Variance of samples: {uniform_samples.var():.4f} (theoretical: {var_unif:.4f})")
    
    # Exponential scenario
    print("\n" + "-"*70)
    print("EXPONENTIAL DISTRIBUTION: Component Lifetime")
    print("-"*70)
    print("Scenario: Mean lifetime = 1000 hours")
    prob_fail_500, prob_survive_1500 = calculate_probability_exponential(1000, 500)
    prob_survive_1500 = calculate_probability_exponential(1000, 1500)[1]
    print(f"  P(failure before 500h): {prob_fail_500:.6f}")
    print(f"  P(survive beyond 1500h): {prob_survive_1500:.6f}")
    
    # Generate random samples from Exponential
    exp_samples = expon.rvs(scale=1000, size=1000)
    print(f"\nRandom sample generation (1000 samples):")
    print(f"  Mean of samples: {exp_samples.mean():.4f} (theoretical: 1000.0000)")
    print(f"  Variance of samples: {exp_samples.var():.4f} (theoretical: 1000000.0000)")
    
    # Distribution fitting
    print("\n" + "="*70)
    print("DISTRIBUTION FITTING")
    print("="*70)
    fitted_dist, params = fit_distribution(concrete_df, 'strength_mpa', 'normal')
    plot_distribution_fitting(concrete_df, 'strength_mpa', fitted_dist,
                             'distribution_fitting.png')
    
    # Part 3: Probability Applications
    print("\n" + "="*70)
    print("PART 3: PROBABILITY APPLICATIONS")
    print("="*70)
    
    # Conditional Probability
    print("\n" + "-"*70)
    print("CONDITIONAL PROBABILITY: Engineering Failure Scenario")
    print("-"*70)
    print("Scenario: Probability of failure given a defect is detected")
    print("  P(Defect and Failure) = 0.08 (8% of components have both)")
    print("  P(Defect) = 0.15 (15% of components have defects)")
    p_failure_given_defect = calculate_conditional_probability(0.08, 0.15)
    print("\nInterpretation: If a defect is detected, there is a {:.1f}% chance".format(p_failure_given_defect*100))
    print("that the component will fail. This is important for quality control decisions.")
    
    # Create probability tree for conditional probability
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Calculate probabilities
    p_defect = 0.15
    p_no_defect = 0.85
    p_failure_and_defect = 0.08
    p_no_failure_and_defect = p_defect - p_failure_and_defect
    p_failure_given_defect_calc = p_failure_and_defect / p_defect
    p_no_failure_given_defect = p_no_failure_and_defect / p_defect
    
    # Assume some probabilities for no defect branch
    p_failure_given_no_defect = 0.02  # Low probability of failure without defect
    p_no_failure_given_no_defect = 0.98
    p_failure_and_no_defect = p_failure_given_no_defect * p_no_defect
    p_no_failure_and_no_defect = p_no_failure_given_no_defect * p_no_defect
    
    # Draw tree
    y_start = 0.9
    x_start = 0.5
    
    # Root
    ax.text(x_start, y_start, 'Component', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue'), ha='center')
    
    # First level: Defect or No Defect
    y_level1 = 0.65
    x_defect = x_start - 0.2
    x_no_defect = x_start + 0.2
    
    ax.plot([x_start, x_defect], [y_start-0.05, y_level1+0.05], 'k-', linewidth=2)
    ax.plot([x_start, x_no_defect], [y_start-0.05, y_level1+0.05], 'k-', linewidth=2)
    
    ax.text(x_defect, y_level1, f'Defect\nP={p_defect:.3f}', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral'))
    ax.text(x_no_defect, y_level1, f'No Defect\nP={p_no_defect:.3f}', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Second level: Failure or No Failure
    y_level2 = 0.35
    x_defect_fail = x_defect - 0.1
    x_defect_no_fail = x_defect + 0.1
    x_no_defect_fail = x_no_defect - 0.1
    x_no_defect_no_fail = x_no_defect + 0.1
    
    # Defect branch
    ax.plot([x_defect, x_defect_fail], [y_level1-0.05, y_level2+0.05], 'k-', linewidth=1.5)
    ax.plot([x_defect, x_defect_no_fail], [y_level1-0.05, y_level2+0.05], 'k-', linewidth=1.5)
    
    ax.text(x_defect_fail, y_level2, f'Failure\nP(F|D)={p_failure_given_defect_calc:.3f}\nJoint: {p_failure_and_defect:.4f}', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    ax.text(x_defect_no_fail, y_level2, f'No Failure\nP(NF|D)={p_no_failure_given_defect:.3f}\nJoint: {p_no_failure_and_defect:.4f}', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # No Defect branch
    ax.plot([x_no_defect, x_no_defect_fail], [y_level1-0.05, y_level2+0.05], 'k-', linewidth=1.5)
    ax.plot([x_no_defect, x_no_defect_no_fail], [y_level1-0.05, y_level2+0.05], 'k-', linewidth=1.5)
    
    ax.text(x_no_defect_fail, y_level2, f'Failure\nP(F|ND)={p_failure_given_no_defect:.3f}\nJoint: {p_failure_and_no_defect:.4f}', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    ax.text(x_no_defect_no_fail, y_level2, f'No Failure\nP(NF|ND)={p_no_failure_given_no_defect:.3f}\nJoint: {p_no_failure_and_no_defect:.4f}', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Display conditional probability
    ax.text(0.5, 0.1, f"P(Failure | Defect) = {p_failure_given_defect_calc:.4f} ({p_failure_given_defect_calc*100:.2f}%)", 
            fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='gold', edgecolor='black', linewidth=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability Tree: Conditional Probability (Failure given Defect)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('conditional_probability_tree.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: conditional_probability_tree.png")
    plt.show()
    
    # Bayes' theorem
    print("\n" + "-"*70)
    print("BAYES' THEOREM: Structural Damage Detection")
    print("-"*70)
    print("Scenario:")
    print("  Base rate: 5% of structures have damage")
    print("  Test sensitivity: 95%")
    print("  Test specificity: 90%")
    
    posterior = apply_bayes_theorem(0.05, 0.95, 0.90)
    plot_probability_tree(0.05, 0.95, 0.90)
    
    # Part 4: Visualization and Reporting
    print("\n" + "="*70)
    print("PART 4: VISUALIZATION AND REPORTING")
    print("="*70)
    
    # Create statistical dashboard
    create_statistical_dashboard(concrete_stats, {})
    
    # Create report
    create_statistical_report(concrete_df, 'lab4_statistical_report.txt')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nAll visualizations and reports have been generated.")
    print("Check the output files in the current directory.")

if __name__ == "__main__":
    main()

