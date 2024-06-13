#!/usr/bin/env python
# coding: utf-8
# Code Courtesy of Caitlin Witt

# In[1]:


import numpy as np
#import enterprise
#from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
import pickle, glob
#import astropy.units as u
import enterprise.constants as const
#import json
#from astropy.coordinates import SkyCoord

#from scipy.stats import norm
import os
from scipy.stats import skewnorm#, truncnorm
from targeted_cws_ng15.jump_proposal import extend_emp_dists

#import matplotlib.pyplot as plt
#from ne2001 import NE2001


# In[20]:


#pickle_dir = 'dist_dm.pkl'
#with open(pickle_dir, 'rb') as f:
    #dist_DM = pickle.load(f)
    #f.close()


# In[21]:


#pickle_dir = 'dist_pi.pkl'
#with open(pickle_dir, 'rb') as f:
    #dist_PX = pickle.load(f)
    #f.close()


# In[11]:


# x = np.linspace(-5,5,1000)
# y = dist_distr2.pdf(x)
# yy = dist_distr_pi_pickle.pdf(x)


# In[19]:


# plt.plot(x,y)
# plt.plot(x,yy)
# plt.xlabel('distance (scaled to 0+/-1)')
# plt.ylabel('prior')


# In[46]:


# dist = 4.6441 
# err = 0.92882

# samps_DM = dist_DM.rvs(size = int(1e6))

# hist = np.histogram(samps_DM*err+dist, bins = 100)
# plt.hist((samps_DM*err+dist), bins = 100);
# plt.axvline(dist, color = 'C1')
# plt.axvline(dist+err, color = 'C1')
# plt.axvline(dist-err, color = 'C1')


# In[22]:

# PXDist from QuickCW, with slight modification to avoid division by zero

'''def Dist_DM_Prior(value):
    """Prior function for ACE SWEPAM parameters."""
    return dist_DM.pdf(value)

def Dist_DM_Sampler(size=None):
    """Sampling function for Uniform parameters."""
    return dist_DM.rvs(size=size)

def Dist_DM_Parameter(size=None):
    """Class factory for ACE SWEPAM parameters."""
    class Dist_DM_Parameter(parameter.Parameter):
        _size = size
        _typename = parameter._argrepr('Dist_DM')
        _prior = parameter.Function(Dist_DM_Prior)
        _sampler = staticmethod(Dist_DM_Sampler)

    return Dist_DM_Parameter'''


# In[47]:



def PXDistPrior(value, dist, err):
    """Prior function for PXDist parameters.

    :param value:   point where we want the prior evaluated
    :param dist:    mean distance
    :param err:     distance error

    :return:        prior value
    """
    
    # this prevents overly large or negative pdist values
    if isinstance(value, float) or isinstance(value, int):
        if value < 0.01:
            return 0
    
    pi = 1/dist
    pi_err = err/dist**2
    
    prior = (1/(np.sqrt(2*np.pi)*pi_err*value**2)*
             np.exp(-(pi-value**(-1))**2/(2*pi_err**2)))
    
    # this prevents overly large or negative pdist values if inputting an array
    if isinstance(prior, np.ndarray):
        prior[value < 0.01] *= 0
    
    return prior

def PXDistSampler(dist, err, size=None):
    """Sampling function for PXDist parameters.

    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter

    :return:        random draw from prior (float or ndarray with lenght size)
    """

    pi = 1/dist
    pi_err = err/dist**2

    #just draw parallax from Gaussian with proper mean and std and return
    #its inverse. But also don't allow values that are too small
    
    return 1/np.random.normal(pi, pi_err)

def PXDistParameter(dist=0, err=1, size=None):
    """Class factory for PX distance parameters with a pdf of inverse Gaussian
    (since parallax is Gaussian)
    
    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter
    
    :return:        ``PXDist`` parameter class
    """

    class PXDist(parameter.Parameter):
        _size = size
        _prior = parameter.Function(PXDistPrior, dist=dist, err=err)
        _sampler = staticmethod(PXDistSampler)
        _typename = parameter._argrepr("PXDist", dist=dist, err=err)

    return PXDist


# In[48]:




class JumpProposalCW(object):

    def __init__(self, pta, fgw=8e-9,psr_dist = None, snames=None, empirical_distr=None, f_stat_file=None):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.ndim = sum(p.size or 1 for p in pta.params)
        self.plist = [p.name for p in pta.params]

        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size

        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct

        # collecting signal parameters across pta
        if snames is None:
            allsigs = np.hstack([[qq.signal_name for qq in pp._signals]
                                                 for pp in pta._signalcollections])
            self.snames = dict.fromkeys(np.unique(allsigs))
            for key in self.snames: self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames: self.snames[key] = list(set(self.snames[key]))
        else:
            self.snames = snames
            
        self.fgw = fgw
        self.psr_dist = psr_dist

        # empirical distributions
        if isinstance(empirical_distr, list):
            # check if a list of emp dists is provided
            self.empirical_distr = empirical_distr

        # check if a directory of empirical dist pkl files are provided
        elif empirical_distr is not None and os.path.isdir(empirical_distr):

            dir_files = glob.glob(empirical_distr+'*.pkl')  # search for pkls

            pickled_distr = np.array([])
            for idx, emp_file in enumerate(dir_files):
                try:
                    with open(emp_file, 'rb') as f:
                        pickled_distr = np.append(pickled_distr, pickle.load(f))
                except:
                    try:
                        with open(emp_file, 'rb') as f:
                            pickled_distr = np.append(pickled_distr, pickle.load(f))
                    except:
                        print(f'\nI can\'t open the empirical distribution pickle file at location {idx} in list!  JumpCW')
                        print("Empirical distributions set to 'None'")
                        pickled_distr = None
                        break

            self.empirical_distr = pickled_distr

        # check if single pkl file provided
        elif empirical_distr is not None and os.path.isfile(empirical_distr):  # checking for single file
            try:
                # try opening the file
                with open(empirical_distr, 'rb') as f:
                    pickled_distr = pickle.load(f)
            except:
                # second attempt at opening the file
                try:
                    with open(empirical_distr, 'rb') as f:
                        pickled_distr = pickle.load(f)
                # if the second attempt fails...
                except:
                    print('\nI can\'t open the empirical distribution pickle file! JumpCW')
                    pickled_distr = None

            self.empirical_distr = pickled_distr

        # all other cases - emp dists set to None
        else:
            self.empirical_distr = None

        if self.empirical_distr is not None:
            # only save the empirical distributions for parameters that are in the model
            mask = []
            for idx,d in enumerate(self.empirical_distr):
                if d.ndim == 1:
                    if d.param_name in pta.param_names:
                        mask.append(idx)
                else:
                    if d.param_names[0] in pta.param_names and d.param_names[1] in pta.param_names:
                        mask.append(idx)
            if len(mask) >= 1:
                self.empirical_distr = [self.empirical_distr[m] for m in mask]
                # extend empirical_distr here:
                print('Extending empirical distributions to priors...\n')
                self.empirical_distr = extend_emp_dists(pta, self.empirical_distr, npoints=100_000)
            else:
                self.empirical_distr = None

        #F-statistic map
        if f_stat_file is not None and os.path.isfile(f_stat_file):
            npzfile = np.load(f_stat_file)
            self.fe_freqs = npzfile['freqs']
            self.fe = npzfile['fe']
    
    def draw_from_many_par_prior(self, par_names, string_name):
        # Preparing and comparing par_names with PTA parameters
        par_names = np.atleast_1d(par_names)
        par_list = []
        name_list = []
        for par_name in par_names:
            pn_list = [n for n in self.plist if par_name in n]
            if pn_list:
                par_list.append(pn_list)
                name_list.append(par_name)
        if not par_list:
            raise UserWarning("No parameter prior match found between {} and PTA.object."
                              .format(par_names))
        par_list = np.concatenate(par_list,axis=None)

        def draw(x, iter, beta):
            """Prior draw function generator for custom par_names.
            par_names: list of strings
            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # randomly choose parameter
            idx_name = np.random.choice(par_list)
            idx = self.plist.index(idx_name)

            # if vector parameter jump in random component
            param = self.params[idx]
            if param.size:
                idx2 = np.random.randint(0, param.size)
                q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            # scalar parameter
            else:
                q[self.pmap[str(param)]] = param.sample()

            # forward-backward jump probability
            lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                    param.get_logpdf(q[self.pmap[str(param)]]))

            return q, float(lqxy)

        name_string = string_name
        draw.__name__ = 'draw_from_{}_prior'.format(name_string)
        return draw
    
    def phase_psi_reverse_jump(self, x, iter, beta):
        ##written by SJV for 11yr CW
        q = x.copy()
        lqxy = 0

        param = np.random.choice([str(p) for p in self.pnames if 'phase' in p])
        
        if param == 'phase0':
            q[self.pnames.index('phase0')] = np.mod(x[self.pnames.index('phase0')] + np.pi, 2*np.pi)
            q[self.pnames.index('psi')] = np.mod(x[self.pnames.index('psi')] + np.pi/2, np.pi)
        else:
            q[self.pnames.index(param)] = np.mod(x[self.pnames.index(param)] + np.pi, 2*np.pi)
                
        return q, float(lqxy)
    
    def fix_cyclic_pars(self, prepar, postpar, iter, beta):
        ##written by SJV for 11yr CW
        q = postpar.copy()
        
        for param in self.params:
            if 'phase' in param.name:
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], 2*np.pi)
            elif param.name == 'psi':
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], np.pi)
            elif param.name == 'gwphi':
                #if param._pmin == 0 and param._pmax == 2*np.pi:
                if str(param).split('=')[1].split(',')[0] == 0 and str(param).split('=')[-1].split(')')[0] == str(2*np.pi):
                    q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], 2*np.pi)
                
        return q, 0

    def fix_psr_dist(self, prepar, postpar, iter, beta):
        ##written by SJV for 11yr CW
        q = postpar.copy()
        
        for param in self.params:
            if 'p_dist' in param.name:
                
                psr_name = param.name.split('_')[0]
                
                while self.psr_dist[psr_name][0] + self.psr_dist[psr_name][1]*q[self.pmap[str(param)]] < 0:
                    q[self.pmap[str(param)]] = param.sample()
                
        return q, 0
    
    def draw_strain_psi(self, x, iter, beta):
        #written by SJV for 11yr CW, adapted for targeted search by CAW
        
        q = x.copy()
        lqxy = 0
        
        # draw a new value of psi, then jump in log10_h so that either h*cos(2*psi) or h*sin(2*psi) are conserved
        which_jump = np.random.random()
        
        if 'log10_h' in self.pnames:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] + np.log10(ratio)       
                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] + np.log10(ratio)
        elif 'log10_fgw' in self.pnames:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])

                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                
            # draw one and calculate the other!!!
            cw_params = [ p for p in self.pnames if p in ['log10_mc', 'log10_fgw']]
            myparam = np.random.choice(cw_params)
            
            idx = 0
            for i,p in enumerate(self.params):
                 if p.name == myparam:
                    idx = i
            param = self.params[idx]
            
            if myparam == 'log10_mc':
                q[self.pnames.index('log10_mc')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_fgw')] = 3/2*(-5/3*q[self.pnames.index('log10_mc')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + np.log10(ratio))

            else:
                q[self.pnames.index('log10_fgw')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_mc')] = 3/5*(-2/3*q[self.pnames.index('log10_fgw')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + np.log10(ratio))
        else:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] + 3/5*np.log10(ratio)       
                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] + 3/5*np.log10(ratio)
                
        return q, float(lqxy)
    
    def draw_strain_inc(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # half of the time, jump so that you conserve h*(1 + cos_inc^2)
        # the rest of the time, jump so that you conserve h*cos_inc
        
        which_jump = np.random.random()
        

        

        if 'log10_h' in self.pnames:
            #written by SJV for 11yr CW, adapted for targeted search (strain not sampled) by CAW

            
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] \
                                                    + np.log10(1+x[self.pnames.index('cos_inc')]**2) \
                                                    - np.log10(1+q[self.pnames.index('cos_inc')]**2)
                        
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] \
                                                    + np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])
        elif 'log10_fgw' in self.pnames:
            
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                ratio =  np.log10(1+x[self.pnames.index('cos_inc')]**2) - np.log10(1+q[self.pnames.index('cos_inc')]**2)

                                               
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                ratio = np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])

            cw_params = [ p for p in self.pnames if p in ['log10_mc', 'log10_fgw']]
            myparam = np.random.choice(cw_params)
            
            idx = 0
            for i,p in enumerate(self.params):
                 if p.name == myparam:
                    idx = i
            param = self.params[idx]
            
            if myparam == 'log10_mc':
                q[self.pnames.index('log10_mc')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_fgw')] = 3/2*(-5/3*q[self.pnames.index('log10_mc')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + ratio)

            else:
                q[self.pnames.index('log10_fgw')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_mc')] = 3/5*(-2/3*q[self.pnames.index('log10_fgw')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + ratio)
                    
            
        else:
        
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] \
                                                    + 3/5*np.log10(1+x[self.pnames.index('cos_inc')]**2) \
                                                    - 3/5*np.log10(1+q[self.pnames.index('cos_inc')]**2)
                        
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] \
                                                    + 3/5*np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])
                    
        return q, float(lqxy)
    
    def draw_strain_skewstep(self, x, iter, beta):
        ##written by SJV for 11yrCW
        
        q = x.copy()
        lqxy = 0
        
        a = 2
        s = 1
        
        diff = skewnorm.rvs(a, scale=s)
        q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] - diff
        lqxy = skewnorm.logpdf(-diff, a, scale=s) - skewnorm.logpdf(diff, a, scale=s)
        
        return q, float(lqxy)
    
    def draw_gwtheta_comb(self, x, iter, beta):
        ##written by SJV for 11yrCW

        q = x.copy()
        lqxy = 0
        
        # the variance of the Gaussian we are drawing from is very small
        # to account for the comb-like structure of the posterior
        sigma = const.c/self.fgw/const.kpc
        
        # now draw an integer to go to a nearby spike
        N = int(0.1/sigma)
        n = np.random.randint(-N,N)
        newval = np.arccos(x[self.pnames.index('cos_gwtheta')]) \
                    + (sigma/2)*np.random.randn() + n*sigma
        
        q[self.pnames.index('cos_gwtheta')] = np.cos(newval)
                
        return q, float(lqxy)

    def draw_gwphi_comb(self, x, iter, beta):
        ##written by SJV for 11yrCW

        # this jump takes into account the comb-like structure of the likelihood 
        # as a function of gwphi, with sharp spikes superimposed on a smoothly-varying function
        # the width of these spikes is related to the GW wavelength
        # this jump does two things:
        #  1. jumps an integer number of GW wavelengths away from the current point
        #  2. draws a step size from a Gaussian with variance equal to half the GW wavelength, 
        #     and takes a small step from its position in a new spike
        
        q = x.copy()
        lqxy = 0
        
        # compute the GW wavelength
        sigma = const.c/self.fgw/const.kpc
        
        # now draw an integer to go to a nearby spike
        # we need to move over a very large number of spikes to move appreciably in gwphi
        # the maximum number of spikes away you can jump 
        # corresponds to moving 0.1 times the prior range
        idx = 0
        for i,p in enumerate(self.params):
            if p.name == 'gwphi':
                idx = i
        pmax = float(str(self.params[idx]).split('=')[-1].split(')')[0])
        pmin = float(str(self.params[idx]).split('=')[1].split(',')[0])
        N = int(0.1*(pmax - pmin)/sigma)

        #N = int(0.1*(self.params[idx]._pmax - self.params[idx]._pmin)/sigma)
        n = np.random.randint(-N,N)
        
        q[self.pnames.index('gwphi')] = x[self.pnames.index('gwphi')] + (sigma/2)*np.random.randn() + n*sigma

        return q, float(lqxy)

# In[ ]:

def group_from_params(pta, params):
    gr = []
    for p in params:
        for q in pta.param_names:
            if p in q:
                gr.append(pta.param_names.index(q))
    return gr

def get_parameter_groups_CAW_target(pta):
    
    """Utility function to get parameter groups for CW sampling.
    These groups should be used instead of the usual get_parameter_groups output.
    Will also include groupings for other signal types for combination with CW signals, if included"""
    
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names

    snames = np.unique([qq.signal_name for pp in pta._signalcollections
                        for qq in pp._signals])
    
    # sort parameters by signal collections
    ephempars = []
    rnpars = []
    cwpars = []
    wnpars = []
    chrompars = []
    gwfspars = [p for p in params if 'crn_log10_rho' in p or 'gwb_log10_rho' in p]

    for sc in pta._signalcollections:
        for signal in sc._signals:
            # note 'red noise' name may also apply to a GWB free spectrum
            if signal.signal_name == 'red noise':
                rnpars.extend(signal.param_names)
            elif signal.signal_name == 'phys_ephem':
                ephempars.extend(signal.param_names)
            elif signal.signal_name == 'cw':
                cwpars.extend(signal.param_names)
            elif signal.signal_name == 'efac':
                wnpars.extend(signal.param_names)
            elif signal.signal_name == 'equad':
                wnpars.extend(signal.param_names)
            elif 'ecorr' in signal.signal_name:
                wnpars.extend(signal.param_names)
            elif signal.signal_name in ['dm_gp', 'chrom_gp', 'sw_r2',
                                        'exp_1', 'exp_2']:
                chrompars.extend(signal.param_names)
    # avoid duplicate red noise param names in case of using free spectral GWB
    rnpars = np.unique(rnpars)

                
    # includes noise and GWB
    if 'red noise' in snames:
        
        # create parameter groups for the red noise parameters
        rnpsrs = [ p.split('_')[0] for p in params if '_log10_A' in p and 'gwb' not in p and 'crn' not in p]
        b = [params.index(p) for p in params if 'alpha' in p]
        for psr in rnpsrs:
            groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')]])
            if 'gwb_log10_A' in params and 'gwb_gamma' in params:
                groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A'), params.index('gwb_gamma'), params.index('gwb_log10_A')]])
            if 'crn_log10_A' in params and 'crn_gamma' in params:
                groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A'), params.index('crn_gamma'), params.index('crn_log10_A')]])
            if len(gwfspars) > 0:
                groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')] + [params.index(p) for p in gwfspars]])

        b = [params.index(p) for p in params if 'alpha' in p]
        groups.extend([b])
        
        for alpha in b:
            groups.extend([[alpha, params.index('J0613-0200_red_noise_gamma'), params.index('J0613-0200_red_noise_log10_A')]])
        
        
        for i in np.arange(0,len(b),2):
            groups.append([b[i],b[i+1]])
        
        
        groups.extend([[params.index(p) for p in rnpars]])
        a = [params.index(p) for p in rnpars]
        if 'log10_fgw' in params:
            a.append(params.index('log10_fgw'))
            groups.extend([a])
            
        a = [params.index(p) for p in rnpars]
        b = []
        if 'gwb_log10_A' in params and 'gwb_gamma' in params:
            b.append(params.index('gwb_log10_A'))
            b.append(params.index('gwb_gamma'))
            if 'gwb_log10_fbend' in params:
                b.append(params.index('gwb_log10_fbend'))
            a.extend(b)
            groups.extend([a])
            groups.extend([b])
            groups.extend([b])
            groups.extend([b])
        if 'crn_log10_A' in params and 'crn_gamma' in params:
            b.append(params.index('crn_log10_A'))
            b.append(params.index('crn_gamma'))
            if 'crn_log10_fbend' in params:
                b.append(params.index('crn_log10_fbend'))
            a.extend(b)
            groups.extend([a])
            groups.extend([b])
            groups.extend([b])
            groups.extend([b])
        if len(gwfspars) > 0:
            for gwpar in gwfspars:
                b.append(params.index(gwpar))
            #a.extend(b) (gwfspars already in rnpars)
            groups.extend([a])
            groups.extend([b])
            groups.extend([b])
            groups.extend([b])
            

    #addition for sampling wn
    #this groups efac and equad together for each pulsar
    if 'efac' in snames and 'equad' in snames:
    
        # create parameter groups for the red noise parameters
        wnpsrs = [p.split('_')[0] for p in params if '_efac' in p]

        for psr in wnpsrs:
            groups.extend([[params.index(psr + '_efac'), params.index(psr + '_log10_equad')]])
            
        groups.extend([[params.index(p) for p in wnpars]])
        
    if 'efac' in snames and 'equad' in snames and 'red noise' in snames:
    
        # create parameter groups for the red noise parameters
        psrs = [p.split('_')[0] for p in params if '_efac' in p and '_log10_A' in p and 'gwb' not in p and 'crn' not in p]

        for psr in psrs:
            groups.extend([[params.index(psr + '_efac'), params.index(psr + '_log10_equad'),
                            params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')]])
            
            
    #create individual parameters groups for chromatic signals:
    for chrom_signal in ['dm_gp', 'chrom_gp', 'sw_r2', 'exp_1', 'exp_2', 'exp1', 'exp2']:
        if chrom_signal in snames:
        
            # create parameter groups for the DMGP parameters
            chrompsrs = np.unique([p.split('_')[0] for p in params if f'_{chrom_signal}' in p])
            
            for psr in chrompsrs:
                chrom_params = pta.signals[f'{psr}_{chrom_signal}'].param_names
                groups.extend([[params.index(par) for par in chrom_params]])
                if f'{psr}_red_noise_log10_A' in params:
                    rn_params = pta.signals[f'{psr}_red_noise'].param_names
                    groups.extend([[params.index(par) for par in chrom_params + rn_params]])
        
    # all chrom params
    groups.extend([[params.index(p) for p in chrompars]])
        
    # make group with all chromatic params + red noise for each pulsar
    for psr in pta.pulsars:
        noise_params = [p for p in chrompars if psr in p]
        if len(noise_params) > 0:
            noise_params += [p for p in rnpars if psr in p]
            groups.extend([[params.index(par) for par in noise_params]])
                    
    # set up groups for the BayesEphem parameters
    if 'phys_ephem' in snames:
        
        ephempars = np.unique(ephempars)
        juporb = [p for p in ephempars if 'jup_orb' in p]
        groups.extend([[params.index(p) for p in ephempars if p not in juporb]])
        groups.extend([[params.index(jp) for jp in juporb]])
        for i1 in range(len(juporb)):
            for i2 in range(i1+1, len(juporb)):
                groups.extend([[params.index(p) for p in [juporb[i1], juporb[i2]]]])
        
    if 'cw' in snames:
        
    
        # divide the cgw parameters into two groups: 
        # the common parameters and the pulsar phase and distance parameters
        cw_common = np.unique(list(filter(lambda x: cwpars.count(x)>1, cwpars)))
        groups.extend([[params.index(cwc) for cwc in cw_common]])

        cw_pulsar = np.array([p for p in cwpars if p not in cw_common])
        if len(cw_pulsar) > 0:
            
            pdist_params = [ p for p in cw_pulsar if 'p_dist' in p ]
            pphase_params = [ p for p in cw_pulsar if 'p_phase' in p ]
            
            for pd,pp in zip(pdist_params,pphase_params):
                if 'cos_gwtheta' in params and 'gwphi' in params:
                    groups.extend([[params.index(pd), params.index('cos_gwtheta'), params.index('gwphi')]])
                groups.extend([[params.index(pd), params.index('log10_mc')]])
                groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc')]])
                groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), 
                                params.index('cos_inc'), params.index('psi')]])
                groups.extend([[params.index(pd), params.index(pp), 
                                params.index('log10_mc')]])
                if 'log10_fgw' in cw_common:
                    groups.extend([[params.index(pd), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_fgw'), 
                                    params.index('cos_inc'), params.index('psi')]])
                    groups.extend([[params.index(pd), params.index(pp), 
                                    params.index('log10_fgw')]])
                    
                    groups.extend([[params.index(pd), params.index(pp), 
                                    params.index('log10_fgw'), params.index('log10_mc')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), 
                                    params.index('cos_inc'), params.index('psi'), params.index('log10_fgw')]])
            
        # now try other combinations of the common cgw parameters
        
        #adapted from get_cw_groups to simplify code
        ang_pars = ['cos_gwtheta', 'gwphi', 'cos_inc', 'phase0', 'psi']
        loc_pars = ['cos_gwtheta', 'gwphi']
        orb_pars = ['cos_inc', 'phase0', 'psi']
        mfdh_pars = ['log10_mc', 'log10_fgw', 'log10_dL', 'log10_h']
        freq_pars = ['log10_mc', 'log10_fgw', 'p_dist', 'p_phase']
        cw_pars = ang_pars.copy()
        cw_pars.extend(mfdh_pars)

        # amp_pars = ['log10_mc', 'log10_h']
        amp_pars = ['log10_mc']
        
        #parameters to catch and match gwb signals - if set to constant or not included, will skip

        crn_pars = ['gwb_gamma', 'gwb_log10_A', 'crn_gamma', 'crn_log10_A']
        crn_cw_pars = crn_pars.copy()
        crn_cw_pars.extend(cw_pars)
        bpl_pars = ['gwb_gamma', 'gwb_log10_A', 'gwb_log10_fbend',
                    'crn_gamma', 'crn_log10_A', 'crn_log10_fbend']
        bpl_cw_pars = bpl_pars.copy()
        bpl_cw_pars.extend(cw_pars)
        
        groups1 = []
        
        for pars in [ang_pars, loc_pars, orb_pars, mfdh_pars, freq_pars, cw_pars, crn_pars, crn_cw_pars, bpl_pars, bpl_cw_pars]:
            if any(item in params for item in pars):
                groups1.append(group_from_params(pta, pars))

        for group in groups1:
            if any(params.index(item) in group for item in amp_pars):
                pass
            else:
                for p in amp_pars:
                    if p in params:
                        g = group.copy()
                        g.append(params.index(p))
                        groups1.append(g)

        groups.extend(groups1)
        
        


    if 'cw' in snames and 'phys_ephem' in snames:
        # add a group that contains the Jupiter orbital elements and the common GW parameters
        juporb = list([p for p in ephempars if 'jup_orb' in p])

        cw_common = list(np.unique(list(filter(lambda x: cwpars.count(x)>1, cwpars))))

        
        myparams = juporb + cw_common
        
        groups.extend([[params.index(p) for p in myparams]])
        
        if 'gwb_log10_A' in params and 'gwb_gamma' in params:
            myparams += ['gwb_log10_A', 'gwb_gamma']
            groups.extend([[params.index(p) for p in myparams]])
        if 'crn_log10_A' in params and 'crn_gamma' in params:
            myparams += ['crn_log10_A', 'crn_gamma']
            groups.extend([[params.index(p) for p in myparams]])
                
    for group in groups:
        if len(group) == 0:
            groups.remove(group)
    return groups

