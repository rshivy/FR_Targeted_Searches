"""
Full targeted search script
"""
import numpy as np
import os
import json
import argparse

from enterprise_extensions.sampler import get_parameter_groups, get_cw_groups, group_from_params, save_runtime_info
from targeted_cws_ng15.jump_proposal import JumpProposal
import targeted_cws_ng15.Dists_Parameters as dists
import enterprise.signals.parameter as parameter

from PTMCMCSampler.PTMCMCSampler import PTSampler as Ptmcmc

from mpi4py import MPI

from tsutils.model_builder import ts_model_builder, ts_broken_powerlaw
from tsutils.utils import get_cw_groups

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


###########
# Options #
###########


def index_format(str_in):
    int_ind = int(str_in)
    index = f'{int_ind:03d}'
    return index


parser = argparse.ArgumentParser()

parser.add_argument('-o', '--out', action='store', type=str, dest='output_suffix')
parser.add_argument('-t', '--target', action='store', type=index_format, dest='target_index')
m_or_h_prior = parser.add_mutually_exclusive_group(required=True)
m_or_h_prior.add_argument('-m', '--mass-prior', action='store', dest='mass_prior',
                          choices=['upper-limit', 'detection'])
m_or_h_prior.add_argument('-s', '--strain-prior', action='store', dest='strain_prior',
                          choices=['upper-limit', 'detection'])
parser.add_argument('-f', '--frequency-prior', action='store', dest='frequency_prior',
                    choices=['constant', 'narrow', 'full'], default='narrow')
parser.add_argument('-b' '--broken-powerlaw', action='store_true', dest='broken_powerlaw')
parser.add_argument('--scratch', action='store_true', dest='scratch')
parser.add_argument('-c', '--hd-correlation', action='store_true', dest='hd_correlation')
parser.add_argument('-y', '--dataset', action='store', dest='dataset', default='20')

args = parser.parse_args()

################
# Data Sources #
################

with open('target_priors_v2/target_index.json', 'r') as f:
    target_prior_fname = json.load(f)[args.target_index]
    target_prior_path = 'target_priors_v2/' + target_prior_fname
    target_name = target_prior_fname[:-5]
if args.dataset == '15':
    psrpath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl'
    noisedict_path = 'noise_dicts/15yr_wn_dict.json'
    psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'
elif args.dataset == '20':
    psrpath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng20_v1p1/v1p1.pkl'
    noisedict_path = 'noise_dicts/ng20_v1p1_dmx_noise_dict.json'
    psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'
else:
    raise ValueError('Dataset must be either 20 for NG 20-year or 15 for NG 15-year')

################
# Setup Output #
################

if args.scratch:
    outputdir = f'/vast/palmer/scratch/mingarelli/frh7/altdata/chains/ng{args.dataset}_v1p1/' + target_name
else:
    outputdir = f'data/chains/ng{args.dataset}_v1p1/' + target_name

if args.mass_prior == 'detection':
    outputdir += '_det'
elif args.mass_prior == 'upper-limit':
    outputdir += '_ul'
elif args.strain_prior == 'detection':
    outputdir += '_hdet'
elif args.strain_prior == 'upper-limit':
    outputdir += '_hul'

if args.frequency_prior == 'narrow':
    outputdir += '_narrowfgw'
elif args.frequency_prior == 'full':
    outputdir += '_varyfgw'
elif args.frequency_prior == 'constant':
    outputdir += '_cfgw'

if args.output_suffix is not None:
    outputdir += '_' + args.output_suffix
if not os.path.exists(outputdir) and rank == 0:
    os.mkdir(outputdir)

if size > 1:
    chainpath = outputdir + '/chain_1.0.txt'
else:
    chainpath = outputdir + '/chain_1.txt'

n_samples = 0
resume = False
if os.path.exists(chainpath):
    with open(chainpath, 'r') as f:
        n_samples = len(f.readlines())
if n_samples >= 100:
    print(f'Continuing from n={n_samples} samples')
    resume = True

##########################
# Setup Enterprise Model #
##########################
if args.broken_powerlaw:
    pta = ts_broken_powerlaw(target_prior_path=target_prior_path,
                             pulsar_path=psrpath,
                             noisedict_path=noisedict_path,
                             pulsar_dists_path=psrdists_path,
                             exclude_pulsars=None,
                             vary_fgw=args.frequency_prior,
                             mass_prior=args.mass_prior)
else:
    pta = ts_model_builder(target_prior_path=target_prior_path,
                           pulsar_path=psrpath,
                           noisedict_path=noisedict_path,
                           pulsar_dists_path=psrdists_path,
                           exclude_pulsars=None,
                           vary_fgw=args.frequency_prior,
                           mass_prior=args.mass_prior,
                           strain_prior=args.strain_prior,
                           hd_correlation=args.hd_correlation)

#######################
# PTMCMCSampler Setup #
#######################


# Get initial sample
x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

# Set up initial covariance matrix
cov = np.diag(np.ones(ndim) * 0.1 ** 2)

# Get parameter groups
if args.hd_correlation:
    gwb_params = ['gwb_log10_A', 'gamma_gwb']
    groups = get_parameter_groups(pta) + get_cw_groups(pta) + [group_from_params(pta, gwb_params)]
else:
    crn_params = ['crn_log10_A', 'gamma_crn']
    groups = get_parameter_groups(pta) + get_cw_groups(pta) + [group_from_params(pta, crn_params)]

# Intialize sampler
sampler = Ptmcmc(ndim=ndim,
                 logl=pta.get_lnlikelihood,
                 logp=pta.get_lnprior,
                 cov=cov,
                 groups=groups,
                 outDir=outputdir,
                 resume=resume)

# Create and add jump proposals
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_red_prior, 30)
sampler.addProposalToCycle(jp.draw_from_cw_prior, 20)
sampler.addProposalToCycle(jp.draw_from_prior, 10)

if args.frequency_prior != 'constant':
    sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_fgw']), 10)
if args.mass_prior:
    sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_mc']), 5)
else:
    sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_h']), 5)
sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 5)

jpCW = dists.JumpProposalCW(pta)
sampler.addProposalToCycle(jpCW.phase_psi_reverse_jump, 1)

pdist_pars = [p for p in pta.param_names if 'p_dist' in p]
pphase_pars = [p for p in pta.param_names if 'p_phase' in p]
sampler.addProposalToCycle(jpCW.draw_from_many_par_prior(pdist_pars, 'p_dist'), 30)
sampler.addProposalToCycle(jpCW.draw_from_many_par_prior(pphase_pars, 'p_phase'), 30)
sampler.addAuxilaryJump(jpCW.fix_cyclic_pars)

###################################
# Save everything before starting #
###################################

if rank == 0:
    # Parsing out the details of the parameters. This might be overly complicated but it works for now
    params = pta.params
    param_names = []
    param_details = {}
    for param in params:
        param_name, _, type_with_args = str(param).partition(':')
        param_type, _, args = type_with_args[:-1].partition('(')
        arg1, _, arg2 = args.partition(', ')
        param_names += [param_name]
        param_details[param_name] = [param_type, arg1, arg2]

    model_parampath = outputdir + '/model_params.json'
    with open(model_parampath, 'w') as f:
        json.dump(param_names, f, indent=4)

    model_priorpath = outputdir + '/model_priors.json'
    with open(model_priorpath, 'w') as f:
        json.dump(param_details, f, indent=4)

    # Also save the constant parameters

    constant_params = set()
    for signal_collection in pta.pulsarmodels:
        for signal in signal_collection.signals:
            for param_name, param in signal._params.items():  # The public attribute only returns nonconstant params
                if isinstance(param, parameter.ConstantParameter):
                    constant_params.add(param)
    constant_params = {cp.name: cp.value if isinstance(cp.value, (np.float64, float))
                       else cp.value.value
                       for cp in sorted(list(constant_params), key=lambda cp: cp.name)}

    constant_parampath = outputdir + '/constant_params.json'
    with open(constant_parampath, 'w') as f:
        json.dump(list(constant_params.keys()), f, indent=4)

    constant_priorpath = outputdir + '/constant_priors.json'
    with open(constant_priorpath, 'w') as f:
        json.dump(constant_params, f, indent=4)

    save_runtime_info(pta=pta,
                      outdir=outputdir,
                      human='Forrest H')

#########
# Begin #
#########

N = 10_000_000
sampler.sample(p0=x0,
               Niter=N,
               SCAMweight=40,
               AMweight=25,
               DEweight=25,
               writeHotChains=True)
