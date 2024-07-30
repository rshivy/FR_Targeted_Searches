import time
from model_builder import ts_model_builder
import numpy as np

if __name__ == '__main__':
    target_prior_path = 'Target_Priors/010_SDSS_J161013.67+311756.4.json'
    psrpath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl'
    noisedict_path = 'noise_dicts/15yr_wn_dict.json'
    psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'

    print('Building PTA...')
    pta = ts_model_builder(target_prior_path=target_prior_path,
                           pulsar_path=psrpath,
                           noisedict_path=noisedict_path,
                           pulsar_dists_path=psrdists_path,
                           exclude_pulsars=None,
                           vary_fgw='narrow')
    print('Sampling')
    t1 = time.perf_counter()
    x = np.hstack([p.sample() for p in pta.params])
    t2 = time.perf_counter()
    print(f'Sampled in {t2 - t1:0.4f} seconds\n')

    print('Evaluating prior')
    t1 = time.perf_counter()
    pta.get_lnprior(x)
    t2 = time.perf_counter()
    print(f'Prior evaluated in {t2 - t1:0.4f} seconds\n')

    print('Evaluating likelihood')
    t1 = time.perf_counter()
    pta.get_lnlikelihood(x)
    t2 = time.perf_counter()
    print(f'Likelihood evaluated in {t2 - t1:0.4f} seconds')