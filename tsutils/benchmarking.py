import time
import numpy as np
from targeted_cws_ng15.models import cw_model_2
from model_builder import ts_model_builder

import json
import pickle

if __name__ == '__main__':
    target_prior_path = 'Target_Priors/010_SDSS_J161013.67+311756.4.json'
    psrpath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl'
    noisedict_path = 'noise_dicts/15yr_wn_dict.json'
    psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'
    ts_model_builder_fixed = ts_model_builder

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