import numpy as np 
import pandas as pd
import pickle 
import os  
import time
from scipy.stats import norm
from itertools import combinations
from statsmodels.api import OLS
from statsmodels.tools import add_constant
from utils import (check_lints, 
                   to_list, 
                   get_combinations, 
                   save_as_pkl, 
                   check_if_exists)


def eba(
        df,
        y,  
        focus=None, 
        free=None, 
        doubtful=None, 
        k=3, 
        model='linear', 
        dropna=True,
        z_score=1.96, 
        verbose=True,
        scale=True,
        max_n=None, 
        savepath = None, 
        save_by_var = False,
        check_for_existing=True,
    ): 
    """Run an EBA analysis on a dataframe.

    Parameters
    ----------
    df : pandas dataframe or numpy array
        Dataframe containing observations
    y : str, list, or numpy array
        Label column name (if exists in df) or vector of labels. 
        If y is a vector, it must have the same length as df.
    focus : list of str or int, optional
        List of variables to study. These are the variables we are 
        interested in. If None, all variables in df are used.
    free : list of str or int, optional
        List of variables to use in every regression. These are 
        assumed to be part of the "true" model and their robustness
        is therefore not checked. If None, no free variables are used.
    doubtful : list of str or int, optional
        For a given focus variable, regression will be run on the free
        variables and all sets of k doubtful variables. If None, all 
        variables in df, except for free variables, are used.
    k : int, optional
        Number of doubtful variables to use in each regression.
    model : str, optional
        Type of model to use. Currently only 'linear' is supported.
    dropna : bool, optional
        If True, drop all rows with missing values during each regression.
    z_score : float, optional
        Z-score to use for confidence intervals for Leamer robustness check 
        (default is 1.96 for 95% confidence intervals). 
    verbose : bool, optional
        If True, print summary of variables used in analysis, and stay updated 
        on progress of analysis.
    scale : bool, optional
        If True, scale all variables to have mean 0 and standard deviation 1.
    max_n : int, optional
        Maximum number of regressions to run for each focus variable. If None,
        all possible regressions are run.
    savepath : str, optional
        Path to save results. If None, results are not saved.
    save_by_var : bool, optional
        If True, save results for each variable separately (if savepath is specified). 
        If savepath is None, this parameter is ignored.
    check_for_existing : bool, optional
        If True, check if results already exist in savepath. If so, skip analysis for 
        these variables. 
        
    Returns
    -------
    results : dict  
        Dictionary containing results of analysis. Keys are names of focus variables. 
        Values are dictionaries, themselves containing: 
            - coef: list of coefficients across all regressions 
            - llf: list of log-likelihoods across all regressions
            - se: list of standard errors across all regressions
            - n: number of regressions run for this variable
            - leamer: Leamer robustness check for this variable (bool)
            - sala_normal: Sala-i-Martin test under normality assumption 
            - sala_generic_lf: Generic Sala-i-Martin test using log-likelihood as weights
            - sala_generic_unweighted: Generic Sala-i-Martin test using equal weights

    """    
    
    # convert df to pandas df if necessary
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
        if not (
            check_lints(focus) and check_lints(free) and check_lints(doubtful)
        ):
            raise ValueError(
                "focus, free, and doubtful must be lists of integers (or None) if df is a numpy array"
                )
        
    # scale data if scale==True 
    if scale:
        df = (df - df.mean()) / df.std() 
        
    # Use all variables as focus and doubtful variables if unspecified
    # Use no variables as free variables if unspecified
    free = [] if free is None else free
    cols_in_play = np.setdiff1d(list(df.columns), free)
    focus = cols_in_play if focus is None else to_list(focus)
    doubtful = cols_in_play if doubtful is None else to_list(doubtful)

    # Summarize variables if verbose 
    if verbose: 
        print('Variables Summary')
        print('-'*60)
        print(f'{len(free)} Free variables: {free}')
        print(f'{len(focus)} Focus variables')
        print(f'{len(doubtful)} Doubtful variables')
        print('-'*60, '\n')
    
    # Add y to df if necessary
    if isinstance(y, str): 
        assert y in list(df.columns), "label string must be in dataframe"
        label_name = y
    elif isinstance(y, list) or isinstance(y, np.ndarray):
        assert len(y) == df.shape[0], "label vector and df must have same number of observations"
        df['y'] = y
        label_name = 'y'    
    
    if model == 'linear':
        model_type = OLS
    else: 
        model_type = OLS

    results = {}
        
    for var in focus: 

        # Skip analysis if existing results are found
        if check_for_existing and check_if_exists(savepath, var):
            if verbose: 
                print(f'Found existing results for: {var}. Skipping.')
            continue

        start = time.time()
        if verbose: 
            print(f'Running analysis for: {var}')
        
        # Get all combinations of doubtful variables for regression
        combs = get_combinations(doubtful, k, [var, label_name], max_n)

        # Run all regressions
        var_results = run_regressions(
            df, var, free, label_name, dropna, model_type, combs
        )
        if var_results['n'] == 0: 
            if verbose: 
                print(f'No regressions able to be run for: {var}. Skipping.')
            continue 

        # Calculate robustness 
        var_results = compute_statistics(var_results, z_score)
        
        # Save these results
        results[var] = var_results
        results[var]['model'] = model
        if savepath is not None and save_by_var:
            save_as_pkl(savepath, var, results[var], verbose=verbose)
    
        if verbose: 
            print(f'Completed analysis for: {var}. ({time.time() - start:.2f} seconds))\n')

    # Save all results
    if savepath is not None: 
        save_as_pkl(savepath, 'all_results', results, verbose=verbose)

    return results 


def run_regressions(df, var, free, label_name, 
                       dropna, model_type, combs):
    """Run all regressions for a single variable"""

    var_results = {
        'coef': [],
        'se': [],
        'llf': [], 
        'n': 0, # number of regressions run
    }

    # Loop over all combinations of k doubtful variables
    for kvars in combs:

        # Exclude combination if it contains the focus variable
        if var in kvars or label_name in kvars:   
            continue 

        data = df[[label_name] + [var] + free + kvars]
        data = data.dropna() if dropna else data
        X = add_constant(data[[var] + free + kvars])
        y = data[label_name]

        # Skip this regression if no observations
        if X.shape[0] == 0: 
            continue
        
        # Run regression
        reg = model_type(y, X).fit()

        # Store coefficients and variance 
        coef = reg.params[var]
        se = reg.bse[var]
        llf = reg.llf
    
        # Store results 
        var_results['coef'].append(coef)
        var_results['se'].append(se)
        var_results['llf'].append(llf)
        var_results['n'] += 1

    return var_results

def compute_statistics(results, z_score):
    

    # Compute weights for each regression based on log-likelihood
    # weights = np.exp(results['llf'])
    weights = results['llf']

    # fix if weights are include nans, are all zeros or all nans
    weights = np.nan_to_num(weights, nan=np.nanmax(weights))
    if np.all(np.isnan(weights)) or np.all(weights == 0) \
        or np.sum(weights) == 0 or np.sum(weights) == np.inf: 
        weights = np.ones(len(weights))

    weights = weights / np.sum(weights)

    # Compute robustness according to various criteria
    results['leamer'] = leamer_check(
        results['coef'], results['se'], z_score
    )
    results['sala_normal'] = sala_check_normal(
        results['coef'], results['se'], weights
    )
    results['sala_generic_lf'] = sala_check_generic(
        results['coef'], results['se'], weights
    )
    results['sala_generic_unweighted'] = sala_check_generic(
        results['coef'], results['se'], None
    )

    return results
    
    
def leamer_check(coef, se, z_score=1.96): 
    """Check if variable with given coefficients and standard errors is 
    robust according to Leamer's criteria, i.e., whether the upper and lower
    bounds of the confidence interval have the same sign. 
    """

    coef = np.array(coef)
    se = np.array(se)

    upper_bounds = coef + z_score * se
    lower_bounds = coef - z_score * se

    return (
        np.all(upper_bounds > 0) and np.all(lower_bounds > 0)\
    ) or \
    (
        np.all(upper_bounds < 0) and np.all(lower_bounds < 0)
    )

def sala_check_normal(coef, se, weights=None): 
    """Check if variable with given coefficients and standard errors is
    robust according to Sala-i-Martin's criteria, i.e., whether the
    coefficient is significantly different from zero.
    """

    if weights is None: 
        weights = np.ones(len(coef))
    
    beta_bar = np.average(coef, weights=weights)
    var_bar = np.sqrt(np.average(np.array(se)**2, weights=weights))

    # Compute cdf evaluated at zero 
    cdf_zero = norm.cdf(0, loc=beta_bar, scale=np.sqrt(var_bar))
    return cdf_zero 


def sala_check_generic(coef, se, weights=None): 

    if weights is None: 
        weights = np.ones(len(coef)) / len(coef) 
    
    cdf_zeros = [
        norm.cdf(0, loc=c, scale=std) for c, std in zip(coef, se)
    ]

    return np.average(cdf_zeros, weights=weights)
    





