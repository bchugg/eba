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

            

