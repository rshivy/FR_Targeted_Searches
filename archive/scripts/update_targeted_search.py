import os
import pickle
import json
import getpass
import numpy as np
import argparse

from targeted_cws_ng15.models import cw_model_2
import targeted_cws_ng15.Dists_Parameters as Dists
from targeted_cws_ng15.jump_proposal import JumpProposal
from PTMCMCSampler.PTMCMCSampler import PTSampler as Ptmcmc
from enterprise_extensions.sampler import save_runtime_info

human = 'F Hutchison'
target_ind = '006'

with open('Target_Priors/target_index.json', 'r') as f:
    source_file_name = json.load(f)[target_ind]
source_name = source_file_name[:-5]
print(f'Target is {source_name}')

outdir = f'data/chains/ng15_v1p1/target_{target_ind}_det'  # set this to wherever you want the output to go

# These are all input sources
# You should have access to my project directory, so you can leave this path as-is
# Or you can copy over the pickle file if you want
datapath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl'
noisedict_path = 'noise_dicts/15yr_wn_dict.json'
psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'
prior_path = f'Target_Priors/{source_file_name}'

Niter = 1_000_000

try:
    os.makedirs(outdir)
except FileExistsError:
    print('Directory already exists')

# This checks if the output file already exists (i.e. this script has already been run).
# This lets you restart if the script was interrupted previously,
# for example if you get preempted on scavenge
try:
    with open(f'{outdir}/chain_1.txt', 'r') as f:
        n_samples = len(f.readlines())  # The number of lines in the file is the
        # number of samples that have already been taken
except FileNotFoundError:
    try:
        with open(f'{outdir}/chain_1.0.txt', 'r') as f:
            n_samples = len(f.readlines())
    except FileNotFoundError:  # If the file doesn't exist then this error is thrown, in which case
        # we want to start from the beginning
        n_samples = 0
if n_samples < 100:
    print(f'{n_samples} samples so far, setting resume = False!')  # If there are only a few then just start over
    resume = False
else:
    print(f'{n_samples} samples so far, resuming')  # If there are > 100 samples already then continue from there
    resume = True

# This loads the pulsar data from a pickle file
# The file holds a list of enterprise Pulsar objects
with open(datapath, 'rb') as f:
    psrs = pickle.load(f)

# Pull the priors for the target parameters (RA, Dec, Freq, mass)
# Although mass is None --> will use log uniform prior
with open(prior_path, 'r') as fin:
    priors = json.load(fin)

# Fixing the red noise parameters to reduce the parameters we are searching over
# And using no spatial correlation (i.e. ignoring Hellings & Downs)
print(f'Using fixed CRN params. ORF = {None}')
log10_A = np.log10(6.4e-15)
gamma = 3.2

# Use a (log-)uniform prior on chirp mass because we have no information
log10_mc_prior = 'uniform'

# This sets up the enterprise PTA object, which integrates the pulsars and their noise models,
# and bundles up all of our parameters.
# The cw_model_2 function is specifically from Bjorn's models.py file, and
# it is relatively comprehensible if you want to check it out
print('run CW model')
pta = cw_model_2(psrs,
                 priors,
                 noisedict_path=noisedict_path,    # Sets the noise parameters for each pulsar from a .json
                 psr_distance_path=psrdists_path,  # Sets the distances to each pulsar from a .pkl
                 orf=None,                         # Use no spatial correlation in the common red noise
                 log10_A_val=log10_A,              # Set the red noise parameters
                 gamma_val=gamma,
                 log10_mc_prior=log10_mc_prior,    # Uniform prior for the chirp mass
                 vary_fgw=False,                   # Use the given frequency, do not vary
                 all_sky=False,                    # Use the given sky coordinates, do not vary
                 fixedpoint=False,
                 nogwb=False)                      # Use the gravitational wave background

# This draws an initial sample for each parameter, giving us the first point in our chain
x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

# This sets the initial covariance matrix. I'm not really sure why this is the value we use,
# but it's a scalar matrix with diagonal entries = 0.01
cov = np.diag(np.ones(ndim) * 0.1 ** 2)

# This sets and records the parameter groups used by the sampler
# I don't really know what is happening here either, gotta ask someone
groups = Dists.get_parameter_groups_CAW_target(pta)
with open(f'{outdir}/groups.txt', 'w') as fi:
    for group in groups:
        line = np.array(pta.param_names)[np.array(group)]
        fi.write("[" + " ".join(line) + "]\n")

# This sets up the sampler
sampler = Ptmcmc(ndim,                  # Number of parameters
                 pta.get_lnlikelihood,  # The (log) likelihood function delivered by the enterprise pta object
                 pta.get_lnprior,       # The (log) priors stored in the enterprise pta object
                 cov,                   # The covariance matrix we made earlier
                 groups=groups,         # The parameter groups
                 outDir=outdir,         # Place to write to
                 resume=resume)         # Whether to start from scratch or pick up where a previous run left off
save_runtime_info(pta,                  # Same stuff, but this function just saves all our info before we start
                  outdir=outdir,
                  human=human)

# This loads a pickle file with the distances to each of the pulsars
with open(psrdists_path, 'rb') as fp:
    dists_file = pickle.load(fp)
psr_dist = {}
for p in psrs: # And this picks out the distances for the pulsars we're actually using
    psr_dist[p.name] = np.array(np.array(dists_file[p.name][:2]))

# This creates a JumpProsals object for our PTA.
# The next 25ish lines call a method of this object to get a particular jump proposal
# in the format excepted by the sampler object.
# These define how the MCMC can update each parameter at each step
# I don't fully understand why each of these is the way it is.
jp = JumpProposal(pta)

# noise prior draws
print('Adding red noise prior draws...')
sampler.addProposalToCycle(jp.draw_from_red_prior, 30)

# Same thing as the regular JumpProposal object but with methods for continuous wave parameters
jpCW = Dists.JumpProposalCW(pta,
                            fgw=10 ** priors['log10_freq'],
                            psr_dist=psr_dist)

# pick a cw param & jump
print('Adding CW prior draws...')
sampler.addProposalToCycle(jp.draw_from_cw_prior, 20)

# draw from Mc
print('Adding Chirp mass prior draws...')
sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_mc']), 5)

print('Adding phase/psi reverse jumps...')
sampler.addProposalToCycle(jpCW.phase_psi_reverse_jump, 1)

# Pulsar term
pdist_pars = [p for p in pta.param_names if 'p_dist' in p]
pphase_pars = [p for p in pta.param_names if 'p_phase' in p]

# draw from p_dists
print('Adding p_dist + p_phase prior draws...')
sampler.addProposalToCycle(jpCW.draw_from_many_par_prior(pdist_pars,
                                                         'p_dist'), 30)

# draw from p_phases
sampler.addProposalToCycle(jpCW.draw_from_many_par_prior(pphase_pars,
                                                         'p_phase'), 30)

print('Adding CW cyclic par auxiliary jump...')
sampler.addAuxilaryJump(jpCW.fix_cyclic_pars)

print('Adding generic prior draws...')
sampler.addProposalToCycle(jp.draw_from_prior, 10)

# saving model paramaters in chain directory
with open(outdir + '/model_params.json', 'w') as fout:
    json.dump(pta.param_names,
              fout,
              sort_keys=True,
              indent=4,                # Makes it print prettier
              separators=(',', ': '))  # This is default

# This does the actual sampling
sampler.sample(x0,              # Initial samples
               Niter,           # Number of samples
               burn=3_000,      # Burn in (how many samples to discard/to get to stationary distribution)
               thin=10,         # What fraction of the samples to save (makes your files smaller)
               SCAMweight=40,   # No idea what this is
               AMweight=25,     # Or this
               DEweight=25)     # Or this
