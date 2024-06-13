#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:03:55 2024

@author: bjornlarsen
"""

import numpy as np
import scipy.integrate as spi
import astropy.units as u

c_light = 299792458 # m/s
G = 6.67430e-11 # Nm^2/kg

def scaled_param_prior(c, param, target_param):
    '''
    Scale a params prior distribution from the one in la_forge
    
    c: la_forge core
    param: str
        Name of param whose prior you are scaling
    target_param: str
        Name of a parameter you are using the scaled prior to solve for. This will determine how to scale the prior
    '''
    # assuming log10_mc prior in units of Msun (for log10_h0 calculation)
    if param == 'log10_mc' and target_param == 'log10_h0':
        pline = [p for p in c.priors if 'log10_mc' in p][0]
        pmin = float(pline[pline.index('pmin')+5:pline.index(', pmax')])
        pmax = float(pline[pline.index('pmax')+5:-1])
        log10_mc_prior = np.array([pmin, pmax])
        log10_mc_prior_scaled = (log10_mc_prior +
                                 np.log10(u.Msun.to(u.kg)*G/c_light**3))
        log10_mc_prior_scaled
        line = [c.runtime_info[i] for i in range(len(c.runtime_info))
                if 'log10_dL' in c.runtime_info[i]][0]
        log10_dL = float(line.replace('log10_dL:Constant=',''))
        log10_dL_scaled = log10_dL + np.log10(u.Mpc.to(u.m)/c_light)
        return 5/3*log10_mc_prior_scaled - log10_dL_scaled + np.log10(2*np.pi)
    # assuming log10_fgw prior in units of 1/s (for log10_h0 calculation)
    elif param == 'log10_fgw' and target_param == 'log10_h0':
        pline = [p for p in c.priors if 'log10_fgw' in p][0]
        pmin = float(pline[pline.index('pmin')+5:pline.index(', pmax')])
        pmax = float(pline[pline.index('pmax')+5:-1])
        log10_fgw_prior = np.array([pmin, pmax])
        return 2/3*log10_fgw_prior
    else:
        raise ValueError('Not sure what scaled prior range you want'
                         f'Param you are trying to scale: {param}'
                         f'Target param: {target_param}'
                         'either need to fix param labelling, or further code development required')

def integrand(x, z, x_prior, y_prior):
    cond1 = (x > x_prior[0])*(x < x_prior[1])
    cond2 = (x > z - y_prior[1])*(x < z - y_prior[0])
    if cond1*cond2:
        return 1/(np.diff(x_prior)[0]*np.diff(y_prior)[0])
    else:
        return 0

def uniform_convolve(z, x_arr, x_prior, y_prior):
    #return spi.quad(integrand, x_arr[0], x_arr[-1], args=(z), limit=200)[0]
    return spi.simpson([integrand(x,z,x_prior,y_prior) for x in x_arr], x=x_arr)#, args=(z))[0]        

def convolved_prior(c, param, Npoints=1000):
    '''
    This point of this function is to numerically calculate the prior for a derived parameter
    which comes from the sum of two parameters sampled using a uniform distribution.
    Unlike the above, this version is agnostic to the convolved parameters and instead
    uses information from the prior directly, as long as the prior string as format:
        'param:Convolve(Uniform(pmin=#, pmax=#), Uniform(pmin=#, pmax=#))'
    Currently only convolution of uniform distributions is supported.
    
    core: la_forge core
        This should ideally contain chains/priors for all 3 params
    param: str
        Name of param whose prior you are solving for
        
    '''
    idx = c.params.index(param)
    pline = c.priors[idx]
    p1_line = pline[pline.index('Convolve(')+len('Convolve('):pline.index(', Uniform')]
    p2_line = pline[pline.index(', Uniform')+2:-1]
    pmin = float(p1_line[p1_line.index('pmin')+5:p1_line.index(', pmax')])
    pmax = float(p1_line[p1_line.index('pmax')+5:-1])
    x_prior = [pmin, pmax]
    pmin = float(p2_line[p2_line.index('pmin')+5:p2_line.index(', pmax')])
    pmax = float(p2_line[p2_line.index('pmax')+5:-1])
    y_prior = [pmin, pmax]
    #x_prior = scaled_param_prior(c, param_x, param_z)
    #y_prior = scaled_param_prior(c, param_y, param_z)
    x_arr = np.linspace(x_prior[0], x_prior[1], Npoints)
    z_arr = np.linspace(x_prior[0] + y_prior[0], x_prior[1] + y_prior[1], Npoints)
    p_z = np.array([uniform_convolve(z, x_arr, x_prior, y_prior) for z in z_arr])
    return z_arr, p_z