import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Tuple

class FrequentistABTesting:
    """
    Class for conducting Frequentist A/B tests (Z-tests for conversion rates,
    T-tests for continuous metrics like revenue).
    """

    @staticmethod
    def z_test_proportions(control_conversions: int, control_trials: int, 
                           treatment_conversions: int, treatment_trials: int) -> Dict[str, Any]:
        """
        Perform a Z-test for proportions, typically used for conversion rate metrics.
        
        :return: A dictionary containing p-value, z-statistic, and confidence intervals.
        """
        p_control = control_conversions / control_trials
        p_treatment = treatment_conversions / treatment_trials
        
        # Pooled conversion rate
        p_pooled = (control_conversions + treatment_conversions) / (control_trials + treatment_trials)
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_trials + 1/treatment_trials))
        
        z_stat = (p_treatment - p_control) / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat))) # Two-sided
        
        # 95% Confidence Interval for the difference in proportions
        se_diff = np.sqrt(p_control * (1 - p_control) / control_trials + 
                          p_treatment * (1 - p_treatment) / treatment_trials)
        z_critical = stats.norm.ppf(0.975) # 1.96 for 95% CI
        
        diff = p_treatment - p_control
        ci_lower = diff - z_critical * se_diff
        ci_upper = diff + z_critical * se_diff
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'control_cr': p_control,
            'treatment_cr': p_treatment,
            'absolute_difference': diff,
            'relative_lift': (diff / p_control) if p_control > 0 else 0,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'is_significant': p_value < 0.05
        }

    @staticmethod
    def t_test_continuous(control_data: np.ndarray, treatment_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform Welch's T-test for continuous data assuming unequal variances.
        """
        # Welch's t-test does not assume equal population variance
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
        
        mean_control = np.mean(control_data)
        mean_treatment = np.mean(treatment_data)
        diff = mean_treatment - mean_control
        
        # Calculate degrees of freedom for Welch-Satterthwaite equation
        v1, v2 = np.var(control_data, ddof=1), np.var(treatment_data, ddof=1)
        n1, n2 = len(control_data), len(treatment_data)
        
        df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        # Confidence intervals
        t_critical = stats.t.ppf(0.975, df)
        se_diff = np.sqrt(v1/n1 + v2/n2)
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'control_mean': mean_control,
            'treatment_mean': mean_treatment,
            'absolute_difference': diff,
            'relative_lift': (diff / mean_control) if mean_control > 0 else 0,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'is_significant': p_value < 0.05
        }
