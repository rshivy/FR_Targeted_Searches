import time
import numpy as np
import json
import pickle
from targeted_cws_ng15.models import cw_model_2
from model_builder import ts_model_builder

import json
import pickle

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from enterprise.signals import gp_signals, white_signals
from enterprise.signals import parameter
from enterprise_extensions.deterministic import CWSignal  # , cw_delay
from targeted_cws_ng15.new_delays_2 import cw_delay_new as cw_delay
from enterprise.signals import selections
from enterprise.signals import signal_base, utils

from targeted_cws_ng15.Dists_Parameters import PXDistParameter
from QuickCW.PulsarDistPriors import DMDistParameter

if __name__ == '__main__':
    def ts_model_builder_fixed(target_prior_path,
                     pulsar_path,
                     noisedict_path,
                     pulsar_dists_path,
                     exclude_pulsars=None,
                     vary_fgw='narrow'):
        """
        Builds a PTA object according to my usual targeted search model choices
    
        :param target_prior_path: Path to a json file containing prior values for the target
        :param pulsar_path: Path to a pkl containing the NANOGrav pulsars
        :param noisedict_path: Path to a json file containing noise parameter valus
        :param pulsar_dists_path: Path to a pkl containing a dict of pulsar distance parameter values
        :param exclude_pulsars: A list of pulsar names to not use, default is None
        :param vary_fgw: Options are {'constant', 'narrow', and 'full'} narrow is log uniform on target value +/- log10(6)
        """
        #####################
        # Set Target Priors #
        #####################
    
        with open(target_prior_path, 'rb') as f:
            target_priors = json.load(f)
    
        target_ra = target_priors['RA']
        target_dec = target_priors['DEC']
        target_log10_dist = target_priors['log10_dist']
        target_log10_freq = target_priors['log10_freq']
    
        target_coords = SkyCoord(target_ra, target_dec)
        target_coords.representation_type = 'physicsspherical'
        target_cos_theta = np.cos(target_coords.theta.to(u.rad))
        target_phi = target_coords.phi.to(u.rad).value
    
        ################
        # Load Pulsars #
        ################
    
        with open(pulsar_path, 'rb') as f:
            psrs = pickle.load(f)
        # Exclude specified pulsars, if any
        if exclude_pulsars is not None:
            psrs = [psr for psr in psrs if psr.name not in exclude_pulsars]
    
        ##########################
        # Setup Enterprise Model #
        ##########################
    
        tmin = [p.toas.min() for p in psrs]
        tmax = [p.toas.max() for p in psrs]
        tspan = np.max(tmax) - np.min(tmin)
        tref = max(tmax)
    
        tm = gp_signals.MarginalizingTimingModel(use_svd=True)
    
        # CW parameters
        cos_gwtheta = parameter.Constant(val=target_cos_theta)('cos_gwtheta')  # position of source
        gwphi = parameter.Constant(val=target_phi)('gwphi')  # position of source
        log10_dist = parameter.Constant(val=target_log10_dist)('log10_dist')  # sistance to source
        if vary_fgw == 'narrow':  # Allow gw frequency to vary by a factor of six in either direction
            target_log10_freq_low = target_log10_freq - np.log10(6)
            target_log10_freq_high = target_log10_freq + np.log10(6)
            log10_fgw = parameter.Uniform(pmin=target_log10_freq_low,
                                          pmax=target_log10_freq_high)('log10_fgw')
        elif vary_fgw == 'full':
            log10_fgw = parameter.Uniform(pmin=np.log10(1/tspan), pmax=-7)('log10_fgw')
        elif vary_fgw == 'constant':
            log10_fgw = parameter.Constant(val=target_log10_freq)('log10_fgw')
        else:
            raise ValueError(f'Unknown value for vary_fgw: {vary_fgw}.'
                             'options are {\'constant\', \'narrow\', \'full\'}')
    
        log10_mc = parameter.Uniform(8, 11)('log10_mc')  # chirp mass of binary
        phase0 = parameter.Uniform(0, 2 * np.pi)('phase0')  # gw phase
        psi = parameter.Uniform(0, np.pi)('psi')  # gw polarization
        cos_inc = parameter.Uniform(-1, 1)('cos_inc')  # inclination of binary with respect to Earth
    
        # White noise
        backend = selections.Selection(selections.by_backend)
        backend_ng = selections.Selection(selections.nanograv_backends)
    
        efac = parameter.Constant()
        log10_equad = parameter.Constant()
        log10_ecorr = parameter.Constant()
        efeq = white_signals.MeasurementNoise(efac=efac,
                                              log10_t2equad=log10_equad,
                                              selection=backend)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=log10_ecorr,
                                            selection=backend_ng)
    
        log10_A = parameter.Uniform(-20, -11)
        gamma = parameter.Uniform(0, 7)
    
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=tspan)
    
        # Common red noise
        log10_A_crn = parameter.Uniform(-18, -11)('crn_log10_A')
        gamma_crn = parameter.Uniform(0, 7)('gamma_crn')
    
        cpl = utils.powerlaw(log10_A=log10_A_crn, gamma=gamma_crn)
    
        crn = gp_signals.FourierBasisGP(cpl, components=14, Tspan=tspan, name='crn')
    
        s = tm + efeq + ec + crn + rn
    
        with open(pulsar_dists_path, 'rb') as f:
            psrdists = pickle.load(f)
        p_phase = parameter.Uniform(0, 2*np.pi)
        signal_collections = []
        # Adding individual cws so we can set pulsar distances
        for psr in psrs:
            # Set pulsar distance parameter
            if psrdists[psr.name][2] == 'PX':
                p_dist = PXDistParameter(dist=psrdists[psr.name][0],
                                         err=psrdists[psr.name][1])
            elif psrdists[psr.name][2] == 'DM':
                p_dist = DMDistParameter(dist=psrdists[psr.name][0],
                                         err=psrdists[psr.name][1])
            else:
                raise ValueError("Can't find if distances are PX or DM!")
    
            cw_wf = cw_delay(cos_gwtheta=cos_gwtheta,
                             gwphi=gwphi,
                             log10_fgw=log10_fgw,
                             log10_mc=log10_mc,
                             phase0=phase0,
                             psi=psi,
                             cos_inc=cos_inc,
                             log10_dist=log10_dist,
                             tref=tref,
                             evolve=True,
                             psrTerm=True,
                             p_dist=p_dist,
                             p_phase=p_phase,
                             scale_shift_pdists=False)  # Bjorn's toggle to fix pulsar distances
    
            cw = CWSignal(cw_wf, ecc=False, psrTerm=True)
            signal_collection = s + cw
            signal_collections += [signal_collection(psr)]
    
        # Instantiate signal collection
        pta = signal_base.PTA(signal_collections)
    
        # Set white noise parameters
        with open(noisedict_path, 'r') as fp:
            noise_params = json.load(fp)
        pta.set_default_params(noise_params)
    
        return pta




    
    target_prior_path = 'Target_Priors/010_SDSS_J161013.67+311756.4.json'
    psrpath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl'
    noisedict_path = 'noise_dicts/15yr_wn_dict.json'
    psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'

    def test_fh_old():
        print('FRH Script\n')
        print('Building PTA...')
        pta = ts_model_builder(target_prior_path=target_prior_path,
                               pulsar_path=psrpath,
                               noisedict_path=noisedict_path,
                               pulsar_dists_path=psrdists_path,
                               exclude_pulsars=None,
                               vary_fgw='constant')
    
        print('Evaluating likelihood')
        np.random.seed(123456789)
        x = np.hstack([p.sample() for p in pta.params])
        pta.get_lnlikelihood(x)
        times = 0
        for i in range(100):
            x = np.hstack([p.sample() for p in pta.params])
            t1 = time.perf_counter()
            pta.get_lnlikelihood(x)
            t2 = time.perf_counter()
            times += t2-t1
        print(f'Average over 100 evaluations: {times/100}')
        fot = times/100
        with open('/vast/palmer/scratch/mingarelli/frh7/benchmark_out/pta1.txt', 'w') as f:
            f.write(pta.summary())
        return fot

    def test_bl():
        print('\nBL Script + pdist fix\n')
        print('Building PTA...')
        with open(target_prior_path, 'r') as f:
            priors = json.load(f)
        with open(psrpath, 'rb') as f:
            psrs = pickle.load(f)
        pta = cw_model_2(psrs, priors, 
                         noisedict_path=noisedict_path,
                         psr_distance_path=psrdists_path,
                         log10_mc_prior='uniform')
    
        print('Evaluating likelihood')
        np.random.seed(123456789)
        x = np.hstack([p.sample() for p in pta.params])
        pta.get_lnlikelihood(x)
        times = 0
        for i in range(100):
            x = np.hstack([p.sample() for p in pta.params])
            t1 = time.perf_counter()
            pta.get_lnlikelihood(x)
            t2 = time.perf_counter()
            times += t2-t1
        print(f'Average over 100 evaluations: {times/100}')
        bt = times/100
        with open('/vast/palmer/scratch/mingarelli/frh7/benchmark_out/pta2.txt', 'w') as f:
            f.write(pta.summary())
        return bt

    def test_fh():
        print('\nFRH Script with no red noise selection + p_phase params + astropy edit\n')
        print('Building PTA...')
        pta = ts_model_builder_fixed(target_prior_path=target_prior_path,
                               pulsar_path=psrpath,
                               noisedict_path=noisedict_path,
                               pulsar_dists_path=psrdists_path,
                               exclude_pulsars=None,
                               vary_fgw='constant')
    
        print('Evaluating likelihood')
        np.random.seed(123456789)
        x = np.hstack([p.sample() for p in pta.params])
        pta.get_lnlikelihood(x)
        times = 0
        for i in range(100):
            x = np.hstack([p.sample() for p in pta.params])
            t1 = time.perf_counter()
            pta.get_lnlikelihood(x)
            t2 = time.perf_counter()
            times += t2-t1
        print(f'Average over 100 evaluations: {times/100}')
        ft = times/100
        with open('/vast/palmer/scratch/mingarelli/frh7/benchmark_out/pta3.txt', 'w') as f:
            f.write(pta.summary())
        return ft

    fot = test_fh_old()
    ft = test_fh()
    bt = test_bl()

    print('PTA summaries written to scratch/benchmark_out')
    print(f'old FRH script takes {(fot-bt)/bt:.1%} longer')
    print(f'revised FRH script takes {(ft-bt)/bt:.1%} longer')