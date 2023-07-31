import numpy as np 
import pandas as pd
import time
import os 
import pickle 
from copy import deepcopy
from statsmodels.api import OLS
from statsmodels.tools import add_constant
from linearmodels.panel import PanelOLS
from joblib import Parallel, delayed
from .utils import (to_list, 
                   get_combinations, 
                   save_as_pkl, 
                   remove_existing, 
                   leamer_check, 
                   sala_check_generic, 
                   sala_check_normal)

class EBA(object): 
    """Class for running Extreme Bounds Analysis on a dataframe.

    Attributes
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
    
    """

    def __init__(
            self, 
            k=3,
            model='linear',
            dropna=True,
            z_score=1.96,
            verbose=True,
            scale=True,
            max_n=None,
            n_proc=1,
            savepath=None,
            save_by_var=False,
            check_for_existing=True,           
            entity_effects=False, 
            time_effects=False
            ) -> None:
        
        self.k = k
        self.model_name = model
        self.dropna = dropna
        self.z_score = z_score
        self.verbose = verbose
        self.scale = scale
        self.max_n = max_n
        self.n_proc = n_proc
        self.savepath = savepath
        self.save_by_var = save_by_var
        self.check_for_existing = check_for_existing
        self.entity_effects = entity_effects
        self.time_effects = time_effects

        # Panel data attributes. These are set during call to run
        self.entity_col = None 
        self.time_col = None
        self.panel = entity_effects or time_effects

        _accepted_models = ['linear']
        self.model = None
        if self.model_name not in _accepted_models:
            raise ValueError(f'{self.model} is not a valid model. Please choose from {_accepted_models}')
        if self.model_name == 'linear' and not self.panel:
            self.model = OLS
        elif self.model_name == 'linear' and self.panel:
            self.model = PanelOLS

    def _prepare_df(self, df, y, entity_col=None, time_col=None):

        self.df = df
        self.entity_col = entity_col
        self.time_col = time_col

        # convert df to pandas df if necessary
        if isinstance(self.df, np.ndarray):
            self.df = pd.DataFrame(self.df)
        
        # Add y to df if necessary
        self.label_name = None
        if isinstance(y, str): 
            assert y in self.df.columns, "label string must be in dataframe"
            self.label_name = y
        elif isinstance(y, list) or isinstance(y, np.ndarray):
            assert len(y) == self.df.shape[0], "label vector and df must have same number of observations"
            self.df['y'] = y
            self.label_name = 'y' 

        # Setup for panel data if necessary 
        if self.panel:
            try: 
                self.df.set_index([self.entity_col, self.time_col], inplace=True)
            except KeyError as e:
                # user already correctly set index
                pass 

        # scale data if scale==True 
        if self.scale:
            self.df = (self.df - self.df.mean()) / self.df.std() 
            

    def _prepare_vars(self, focus, free, doubtful):

        # Use all variables as focus and doubtful variables if unspecified
        # Use no variables as free variables if unspecified
        self.free = [] if free is None else free
        feature_cols = deepcopy(list(self.df.columns))
        feature_cols.remove(self.label_name)
        cols_in_play = np.setdiff1d(feature_cols, self.free)
        self.focus = cols_in_play if focus is None else to_list(focus)
        self.doubtful = cols_in_play if doubtful is None else to_list(doubtful)

        # Ensure that focus and doubtful variables are in df
        for var in np.append(self.focus, self.doubtful):
            assert var in self.df.columns, f'{var} is not in dataframe'
    
    
    def run(self, df, y, focus=None, free=None, doubtful=None, 
            entity_col=None, time_col=None): 

        
        self._prepare_df(df, y, entity_col, time_col)
        self._prepare_vars(focus, free, doubtful)

        # Summarize variables if verbose 
        if self.verbose: 
            print('Variables Summary')
            print('-'*60)
            print(f'{len(self.free)} Free variables: {self.free}')
            print(f'{len(self.focus)} Focus variables')
            print(f'{len(self.doubtful)} Doubtful variables')
            print('-'*60, '\n')
        
        # Skip previous analyses if found 
        if self.check_for_existing: 
            self.focus = remove_existing(self.focus, self.savepath, self.verbose)

        # Run analysis for each free variable
        results_dict = self._run_regressions(
            get_combinations(self.doubtful, self.k, [self.label_name], self.max_n)
        )
        for var, res in results_dict.items():
            res = self._compute_statistics(res)
            res['model'] = self.model_name

        # Run analysis for each focus variable 
        results_focus = Parallel(n_jobs=self.n_proc)(
            delayed(self._process_var)(var) for var in self.focus
        )
        for res, var in zip(results_focus, self.focus):  
            results_dict[var] = res

        # Save all results
        if self.savepath is not None: 
            save_as_pkl(self.savepath, 'all_results', 
                        results_dict, verbose=self.verbose)

        return results_dict
    
    def _process_var(self, var): 

        # Check if results already exist for this variable
        if self.savepath is not None and self.check_for_existing: 
            if os.path.exists(os.path.join(self.savepath, f'{var}.pkl')):
                with open(os.path.join(self.savepath, f'{var}.pkl'), 'rb') as f:
                    results = pickle.load(f)
                    return results             

        start = time.time()
        if self.verbose: 
            print(f'Running analysis for: {var}')
        
        # Get all combinations of doubtful variables for regression
        combs = get_combinations(self.doubtful, self.k, [var, self.label_name], self.max_n)

        # Run all regressions
        var_results = self._run_regressions(combs, var)[var]
        if var_results['n'] == 0: 
            if self.verbose: 
                print(f'No regressions able to be run for: {var}. Skipping.')
            return var_results

        # Calculate robustness 
        var_results = self._compute_statistics(var_results)
        
        # Save these results
        var_results['model'] = self.model_name
        if self.savepath is not None and self.save_by_var:
            save_as_pkl(self.savepath, var, var_results, verbose=self.verbose)

        if self.verbose: 
            print(f'Completed analysis for: {var}. ({time.time() - start:.2f} seconds))\n')

        return var_results

        
    def _run_regressions(self, combs, var=None): 

        vars_to_track = self.free if var is None else [var]
        results = {free: {
            'coef': [],
            'se': [],
            'llf': [],
            'n': 0,
            'n_obs': []
        } for free in vars_to_track}

        for kvars in combs: 

            reg_vars = kvars if var is None else [var] + kvars
            reg_results, n_obs = self._run_single_regression(reg_vars)

            # Update results
            for key, val in results.items():

                # Get and store results 
                if self.model == OLS: 
                    coef = reg_results.params[key]
                    se = reg_results.bse[key]
                    llf = reg_results.llf
                elif self.model == PanelOLS: 
                    coef = reg_results.params[key]
                    se = reg_results.std_errors[key]
                    llf = reg_results.loglik
        
                val['coef'].append(coef)
                val['se'].append(se)
                val['llf'].append(llf)
                val['n'] += 1
                val['n_obs'].append(n_obs)

        # Save 
        if self.savepath is not None: 
            save_as_pkl(self.savepath, 'base_vars', 
                    results, verbose=self.verbose)
            
        return results

    def _run_single_regression(self, included_vars): 

        data = self.df[[self.label_name] + self.free + included_vars]
        data = data.dropna() if self.dropna else data
        if not self.panel and self.model_name == 'linear':
            # only add constant if not deadling with fixed-effects via Panel OLS 
            X = add_constant(data[self.free + included_vars])
        else: 
            X = data[self.free + included_vars]
        y = data[self.label_name]
        
        # Skip this regression if no observations
        n_obs = X.shape[0]
        if n_obs == 0: 
            return None, 0 

        reg = None
        if self.model_name == 'linear':  
            if self.panel: 
                reg = PanelOLS(y, X, entity_effects=self.entity_effects, 
                            time_effects=self.time_effects).fit()
            else: 
                reg = OLS(y, X).fit()

        return reg, n_obs
        
            
    
    # def _run_regressions(self, var, combs):
    #     """Run all regressions for a single variable"""

    #     var_results = {
    #         'coef': [],
    #         'se': [],
    #         'llf': [], 
    #         'n': 0, # number of regressions run
    #         'n_obs': [], # number of observations in each regression
    #     }

    #     # Loop over all combinations of k doubtful variables
    #     for kvars in combs:

    #         # Run regression
    #         reg_results = self._run_single_regression([var] + kvars)


    #         # Get and store results 
    #         if self.model == OLS: 
    #             coef = reg_results.params[var]
    #             se = reg_results.bse[var]
    #             llf = reg_results.llf
    #         elif self.model == PanelOLS: 
    #             coef = reg_results.params[var]
    #             se = reg_results.std_errors[var]
    #             llf = reg_results.loglik
            
    #         var_results['coef'].append(coef)
    #         var_results['se'].append(se)
    #         var_results['llf'].append(llf)
    #         var_results['n'] += 1
    #         # var_results['n_obs'].append(n_obs)

    #     return var_results

    def _compute_statistics(self, results):
        

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
            results['coef'], results['se'], self.z_score
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
         
    