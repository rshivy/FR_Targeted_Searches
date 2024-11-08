import numpy as np
#import glob, json
#import matplotlib.pyplot as plt
#from astropy import units as u
#from astropy.coordinates import SkyCoord
#import scipy as sp
#import os
#import pickle as pickle

#from enterprise.signals import parameter
#from enterprise.pulsar import Pulsar
#from enterprise.signals import selections
from enterprise.signals import signal_base
#from enterprise.signals import white_signals
#from enterprise.signals import gp_signals
#from enterprise.signals import deterministic_signals
import enterprise.constants as const
#from enterprise.signals import utils

#from enterprise_extensions.models import t_process
#from enterprise_extensions.models import (InvGamma, InvGammaPrior,
#                                          InvGammaSampler)

#from enterprise_extensions import model_utils

def create_gw_antenna_pattern_new(pos, gwtheta, gwphi, gwpsi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param theta: Polar angle of pulsar location.
    :param phi: Azimuthal angle of pulsar location.
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians
    
    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the 
             pulsar and the GW source.
    """
    """
    JAE version:
        m = np.array([np.sin(gwphi), -np.cos(gwphi), 0.0])
        n = np.array([-np.cos(gwtheta)*np.cos(gwphi), 
                  -np.cos(gwtheta)*np.sin(gwphi),
                  np.sin(gwtheta)])
        omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi), 
                      -np.sin(gwtheta)*np.sin(gwphi),
                      -np.cos(gwtheta)])
    """
    # use definition from SRT
    
    
    
    n = np.array([np.sin(gwtheta)*np.cos(gwphi),
                 np.sin(gwtheta)*np.sin(gwphi),
                 np.cos(gwtheta)])
    omhat = -n
    p = np.array([np.cos(gwpsi)*np.cos(gwtheta)*np.cos(gwphi)-
                  np.sin(gwpsi)*np.sin(gwphi),
                 np.cos(gwpsi)*np.cos(gwtheta)*np.sin(gwphi)+
                 np.sin(gwpsi)*np.cos(gwphi),
                 -np.cos(gwpsi)*np.sin(gwtheta)])
    #p = -n for psi = 0
    
    q = np.array([np.sin(gwpsi)*np.cos(gwtheta)*np.cos(gwphi)+
                  np.cos(gwpsi)*np.sin(gwphi),
                 np.sin(gwpsi)*np.cos(gwtheta)*np.sin(gwphi)-
                 np.cos(gwpsi)*np.cos(gwphi),
                 -np.sin(gwpsi)*np.sin(gwtheta)])
    #q = m for psi = 0

    u = pos #np.array([np.sin(theta)*np.cos(phi), 
                     #np.sin(theta)*np.sin(phi), 
                     #np.cos(theta)])

    fplus = 0.5 * (np.dot(u,p)**2 - np.dot(u, q)**2) / (1+np.dot(omhat, u))
    fcross = (np.dot(u, p)*np.dot(u, q)) / (1 + np.dot(omhat, u))
    cosMu = -np.dot(omhat, u)

    return fplus, fcross, cosMu

@signal_base.function
def cw_delay_new(toas, pos, pdist,
                 cos_gwtheta=0, gwphi=0, cos_inc=0,
                 log10_mc=9, log10_fgw=-8, log10_dist=None, log10_h=None,
                 phase0=0, psi=0,
                 psrTerm=False, p_dist=1, p_phase=None,
                 evolve=False, phase_approx=False, check=False,
                 tref=0, scale_shift_pdists=True):
    """
    Function to create GW incuced residuals from a SMBMB as
    defined in Ellis et. al 2012,2013.
    :param toas:
        Pular toas in seconds
    :param pos:
        Unit vector from the Earth to the pulsar
    :param pdist:
        Pulsar distance (mean and uncertainty) [kpc]
    :param cos_gwtheta:
        Cosine of Polar angle of GW source in celestial coords [radians]
    :param gwphi:
        Azimuthal angle of GW source in celestial coords [radians]
    :param cos_inc:
        cosine of Inclination of GW source [radians]
    :param log10_mc:
        log10 of Chirp mass of SMBMB [solar masses]
    :param log10_fgw:
        log10 of Frequency of GW (twice the orbital frequency) [Hz]
    :param log10_dist:
        log10 of Luminosity distance to SMBMB [Mpc],
        used to compute strain, if not None
    :param log10_h:
        log10 of GW strain,
        used to compute distance, if not None
    :param phase0:
        Initial Phase of GW source [radians]
    :param psi:
        Polarization angle of GW source [radians]
    :param psrTerm:
        Option to include pulsar term [boolean]
    :param p_dist:
        Pulsar distance parameter
    :param p_phase:
        Use pulsar phase to determine distance [radian]
    :param evolve:
        Option to include/exclude full evolution [boolean]
    :param phase_approx:
        Option to include/exclude phase evolution across observation time
        [boolean]
    :param check:
        Check if frequency evolves significantly over obs. time [boolean]
    :param tref:
        Reference time for phase and frequency [s]
    :param scale_shift_pdists:
        Toggle to scale and shift input p_dist param by distances in
        enterprise pulsar. False to just use the input p_dist param directly.
        Default True [boolean]
    :return: Vector of induced residuals
    """
    # convert units to time
    mc = 10 ** log10_mc * const.Tsun
    fgw = 10 ** log10_fgw
    gwtheta = np.arccos(cos_gwtheta)
    inc = np.arccos(cos_inc)
    if scale_shift_pdists:
        p_dist = (pdist[0] + pdist[1] * p_dist) * const.kpc / const.c
    else:
        p_dist = p_dist * const.kpc / const.c

    if log10_h is None and log10_dist is None:
        raise ValueError("one of log10_dist or log10_h must be non-None")
    elif log10_h is not None and log10_dist is not None:
        raise ValueError("only one of log10_dist or log10_h can be non-None")
    elif log10_h is None:
        dist = 10 ** log10_dist * const.Mpc / const.c
    else:
        dist = 2 * mc ** (5 / 3) * (np.pi * fgw) ** (2 / 3) / 10 ** log10_h

    if check:
        # check that frequency is not evolving significantly over obs. time
        fstart = fgw * (1 - 256 / 5 * mc ** (5 / 3) * fgw ** (8 / 3) * toas[0]) ** (-3 / 8)
        fend = fgw * (1 - 256 / 5 * mc ** (5 / 3) * fgw ** (8 / 3) * toas[-1]) ** (-3 / 8)
        df = fend - fstart

        # observation time
        Tobs = toas.max() - toas.min()
        fbin = 1 / Tobs

        if np.abs(df) > fbin:
            print('WARNING: Frequency is evolving over more than one '
                  'frequency bin.')
            print('f0 = {0}, f1 = {1}, df = {2}, fbin = {3}'
                  .format(fstart, fend, df, fbin))
            return np.ones(len(toas)) * np.nan

    # get antenna pattern funcs and cosMu
    # write function to get pos from theta,phi
    fplus, fcross, cosMu = create_gw_antenna_pattern_new(pos, gwtheta,
                                                         gwphi, psi)

    # get pulsar time

    # toas -= tref
    # safety! compounds can go wrong!
    toas = toas - tref
    if p_dist > 0:
        tp = toas - p_dist * (1 - cosMu)
    else:
        tp = toas

    # orbital frequency
    w0 = np.pi * fgw
    phase0 /= 2  # orbital phase
    # omegadot = 96/5 * mc**(5/3) * w0**(11/3)

    # evolution
    if evolve:
        # calculate time dependent frequency at earth and pulsar
        omega = w0 * (1 - 256 / 5 * mc ** (5 / 3) * w0 ** (8 / 3) * toas) ** (-3 / 8)
        omega_p = w0 * (1 - 256 / 5 * mc ** (5 / 3) * w0 ** (8 / 3) * tp) ** (-3 / 8)

        if p_dist > 0:
            omega_p0 = w0 * (1 + 256 / 5
                             * mc ** (5 / 3) * w0 ** (8 / 3) *
                             p_dist * (1 - cosMu)) ** (-3 / 8)
        else:
            omega_p0 = w0

        # calculate time dependent phase
        phase = phase0 + 1 / 32 / mc ** (5 / 3) * (w0 ** (-5 / 3) - omega ** (-5 / 3))

        if p_phase is None:
            phase_p = phase0 + 1 / 32 / mc ** (5 / 3) * (w0 ** (-5 / 3) -
                                                         omega_p ** (-5 / 3))
        else:
            phase_p = (phase0 + p_phase
                       + 1 / 32 * mc ** (-5 / 3) * (omega_p0 ** (-5 / 3) -
                                                    omega_p ** (-5 / 3)))

    elif phase_approx:
        # monochromatic
        omega = w0
        if p_dist > 0:
            omega_p = w0 * (1 + 256 / 5
                            * mc ** (5 / 3) * w0 ** (8 / 3) * p_dist * (1 - cosMu)) ** (-3 / 8)
        else:
            omega_p = w0

        # phases
        phase = phase0 + omega * toas
        if p_phase is not None:
            phase_p = phase0 + p_phase + omega_p * toas
        else:
            phase_p = (phase0 + omega_p * toas
                       + 1 / 32 / mc ** (5 / 3) * (w0 ** (-5 / 3) - omega_p ** (-5 / 3)))

    # no evolution
    else:
        # monochromatic
        omega = np.pi * fgw
        omega_p = omega

        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + omega * tp

    # define time dependent coefficients
    At = -0.5 * np.sin(2 * phase) * (3 + np.cos(2 * inc))
    Bt = 2 * np.cos(2 * phase) * np.cos(inc)
    At_p = -0.5 * np.sin(2 * phase_p) * (3 + np.cos(2 * inc))
    Bt_p = 2 * np.cos(2 * phase_p) * np.cos(inc)

    # now define time dependent amplitudes
    alpha = mc ** (5. / 3.) / (dist * omega ** (1. / 3.))
    alpha_p = mc ** (5. / 3.) / (dist * omega_p ** (1. / 3.))

    # define rplus and rcross
    '''
    rplus = alpha*(-At*np.cos(2*psi)+Bt*np.sin(2*psi))
    rcross = alpha*(At*np.sin(2*psi)+Bt*np.cos(2*psi))
    rplus_p = alpha_p*(-At_p*np.cos(2*psi)+Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(At_p*np.sin(2*psi)+Bt_p*np.cos(2*psi))
    '''

    rplus = alpha * (At)
    rcross = alpha * (-Bt)
    rplus_p = alpha_p * (At_p)
    rcross_p = alpha_p * (-Bt_p)

    # residuals
    if psrTerm:
        res = fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross)
    else:
        res = -fplus * rplus - fcross * rcross

    return res

