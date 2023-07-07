from itertools import combinations
import numpy as np
import os 
import pickle

def check_lints(arr): 
    """check given array is all integers or None type"""
    return arr is None or all(isinstance(x, int) for x in arr)

def to_list(val): 
    """Convert val to list if not already a list"""
    if not isinstance(val, list): 
        return [val]
    return val

def get_combinations(arr, k, exclude=None, max_n=None): 
    """Get all combinations of k distinct elements from arr. 
    Remove elements from combs if they are in exclude"""
    combs = [list(c) for c in list(combinations(arr, k))]
    if exclude is not None:
        combs = list([c for c in combs if not any([e in c for e in exclude])]) 
    if max_n is not None and len(combs) > max_n:
        np.random.shuffle(combs)
        combs = combs[:max_n]
        
    return combs

def save_as_pkl(savepath, fname, tosave, verbose=True): 
    """Save tosave as fname.pkl in savepath"""

    # strip spaces from fname 
    fname = str(fname).replace(' ', '_')

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    var_savepath = os.path.join(savepath , f'{fname}.pkl')
    with open(var_savepath, 'wb') as f:
        pickle.dump(tosave, f)

    if verbose: 
        print(f'Saved {fname} to {savepath}')

def check_if_exists(savepath, fname): 
    """Check if fname exists in savepath"""
    fname = str(fname).replace(' ', '_')
    var_savepath = os.path.join(savepath , f'{fname}.pkl')
    return os.path.exists(var_savepath)

def remove_existing(vars, savepath, verbose): 
    """Remove vars from list if associated pkl file already exist in savepath"""
    print(vars, savepath, verbose)
    to_skip = []
    for var in vars: 
        if check_if_exists(savepath, var):
            if verbose: 
                print(f'Found existing results for: {var}. Skipping.')
            to_skip.append(var)
    vars = np.setdiff1d(vars, to_skip)
    return vars

            
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

        