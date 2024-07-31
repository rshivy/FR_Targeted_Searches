"""
Full targeted search script
"""
import numpy as np
import os
import json
import argparse

from enterprise_extensions.sampler import get_parameter_groups, save_runtime_info  # , JumpProposal
from targeted_cws_ng15.jump_proposal import JumpProposal
import targeted_cws_ng15.Dists_Parameters as dists
import enterprise.signals.parameter as parameter

from PTMCMCSampler.PTMCMCSampler import PTSampler as Ptmcmc

from mpi4py import MPI

from tsutils.model_builder import ts_model_builder

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

mass_prior = parser.add_mutually_exclusive_group(required=True)
mass_prior.add_argument('-u', '--upper-limit', action='store_true', dest='upper_limit')
mass_prior.add_argument('-d', '--detection', action='store_true', dest='detection')

parser.add_argument('-f', '--frequency-prior', action='store', type=str, dest='frequency_prior',
                    choices=['constant', 'narrow', 'full'], default='narrow')

args = parser.parse_args()

################
# Data Sources #
################

with open('Target_Priors/target_index.json', 'r') as f:
    target_prior_fname = json.load(f)[args.target_index]
    target_prior_path = 'Target_Priors/' + target_prior_fname
    target_name = target_prior_fname[:-5]
psrpath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl'
noisedict_path = 'noise_dicts/15yr_wn_dict.json'
psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'

################
# Setup Output #
################

outputdir = 'data/chains/ng15_v1p1/' + target_name

if args.detection:
    outputdir += '_det'
elif args.upper_limit:
    outputdir += '_ul'

if args.frequency_prior == 'narrow':
    outputdir += '_narrowfgw'
elif args.frequency_prior == 'full':
    outputdir += '_varyfgw'

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

pta = ts_model_builder(target_prior_path=target_prior_path,
                       pulsar_path=psrpath,
                       noisedict_path=noisedict_path,
                       pulsar_dists_path=psrdists_path,
                       exclude_pulsars=None,
                       vary_fgw=args.frequency_prior,
                       mass_prior=args.mass_prior)

#######################
# PTMCMCSampler Setup #
#######################


# Get initial sample
x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

# Set up initial covariance matrix
cov = np.diag(np.ones(ndim) * 0.1 ** 2)

# Get parameter groups
groups = get_parameter_groups(pta)

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

sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_fgw']), 10)
sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_mc']), 5)
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
