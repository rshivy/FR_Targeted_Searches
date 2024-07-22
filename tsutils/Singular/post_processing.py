from pathlib import Path
import numpy as np
from astropy.constants import c, G, M_sun, pc
import json
import matplotlib.pyplot as plt
from la_forge.core import Core
from ..utils import *

# ----------------------------- #
# ---- PostProcessor Object --- #
# ----------------------------- #


class PostProcessor(Core):
    def __init__(self,
                 chaindir=None,
                 corepath=None,
                 burn=0.25,
                 label=None,
                 fancy_par_names=None,
                 chain=None,
                 params=None,
                 pt_chains=False,
                 skiprows=0,
                 usecols=None,
                 true_vals=None):

        super().__init__(chaindir,
                         corepath,
                         burn,
                         label,
                         fancy_par_names,
                         chain,
                         params,
                         pt_chains,
                         skiprows,
                         usecols,
                         true_vals)
        self.datapath = Path(dpath).resolve()
        self._burn_frac = burn


            # Load priors
            print('Loading priors')
            self.priors = load_priors(self.datapath)
            self.constant_priors = constant_priors

        # However the chains were loaded combine them
        full_len = sum([x.shape[0] for x in self.separate_chains.values()])
        full_param = [x.shape[1] for x in self.separate_chains.values()][0]
        print(f'combined chain has '
              f'{full_len} samples with '
              f'{full_param} parameters')

        # This assignment creates the burned chain attribute self.combined_chain_burn
        self.burn = self._burn_frac

    def __getitem__(self, param_name):
        idx = self.params.index(param_name)
        return self.combined_chain_burn[:, idx]

    def plot_trace(self, param_name, ax=None, **kwargs):
        """
        Plots the trace of the given parameter after the currently set burn in

        :param param_name: Parameter to plot the trace of
        :param ax: An optional Axes object to draw the plot on, if None a new one is created.
        :param kwargs: Additional arguments to pass to ax.plot()
        :return: A matplotlib Axes object with the trace plot
        """
        pchain = self[param_name]
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(pchain, **kwargs)
        return ax

    @property
    def burn(self):
        return self._burn_frac

    @burn.setter
    def burn(self, new_burn_frac):
        if hasattr(self, 'combined_chain_burn'):  # Free up precious memory before we make a new big array
            del self.combined_chain_burn
        self._burn_frac = new_burn_frac
        self._burn_nums = {chainkey: int(len(chain) * self._burn_frac)
                           for chainkey, chain
                           in self.separate_chains.items()}
        self.combined_chain_burn = np.vstack([chain[self._burn_nums[chainkey]:]
                                              for chainkey, chain
                                              in self.separate_chains.items()])
        print(f'Burn in fraction is {self.burn}, with {self.combined_chain_burn.shape[0]} samples post burn in')

    @property
    def chain(self):
        return np.vstack([chain for _, chain in sorted(self.separate_chains.items())])

    def add_strain_param(self):
        # This doesn't work yet
        # frequency and distance are as yet undefined
        # And accessing parameters with the [param] syntax doesn't work for the individual chains (need to get index)
        if 'log10_freq' in self.params:
            log10_freq = self['log10_freq']
        elif 'log10_freq' in self.constant_priors.keys():
            log10_freq = self.constant_priors['log10_freq']
        else:
            raise RuntimeError('log10_freq parameter must be in parameter list or in constant parameter dict')

        if 'log10_dist' in self.params:
            log10_dist = self['log10_dist']
        elif 'log10_dist' in self.constant_priors.keys():
            log10_dist = self.constant_priors['log10_dist']
        else:
            raise RuntimeError('log10_dist parameter must be in parameter list or in constant parameter dict')

        log10_mc_idx = self.params.index('log10_mc')
        self.params += ['log10_h']
        separates = {corekey: calc_log10_strain(core[log10_mc_idx], log10_freq, log10_dist).reshape((len(core), 1))
                     for corekey, core in self.separate_chains.items()}
        self.separate_chains = {corekey: np.hstack((core, separates[corekey]))
                                for corekey, core
                                in self.separate_chains.items()}