import numpy as np
from astropy.constants import c, G, M_sun, pc
from enterprise_extensions.sampler import group_from_params


def cg1_mass(solarmass_mass):  # Convert mass from solar mass to seconds
    return solarmass_mass * M_sun.value * (G.value / (c.value ** 3))


def log_cg1_mass(log_solarmass_mass):
    return log_solarmass_mass + np.log10(M_sun.value) + np.log10(G.value / (c.value ** 3))


def cg1_dist(mpc_dist):  # Convert distance from Kpc to seconds
    mpc = pc * (10 ** 6)
    return mpc_dist * mpc.value / c.value


def log_cg1_dist(log_mpc_dist):
    mpc = pc * (10 ** 6)
    return log_mpc_dist + np.log10(mpc.value / c.value)


def calc_strain(solarmass_mass, freq, mpc_dist):
    mass = cg1_mass(solarmass_mass)
    dist = cg1_dist(mpc_dist)
    nom = 2 * (mass ** (5 / 3)) * np.pi * (freq ** (2 / 3))
    return nom / dist


def calc_log10_strain(log10solarmass_mass, log10freq, log10mpc_dist):
    log10mass = log_cg1_mass(log10solarmass_mass)
    log10dist = log_cg1_dist(log10mpc_dist)
    log10strain = (5 / 3) * log10mass + (2 / 3) * log10freq - log10dist + np.log10(2 * np.pi)
    return log10strain


def load_priors(path_to_chain_dir):
    priorpath = path_to_chain_dir / 'priors.txt'
    priorlist = np.loadtxt(priorpath, dtype='str', delimiter='\t')
    priors = {}
    for pline in priorlist:
        ind1 = pline.index(':')
        ind2 = pline.index('(')
        ind3 = pline.index(',')
        ind4 = pline.index(')')

        pname = pline[:ind1]
        ptype = pline[ind1 + 1:ind2]
        parg1 = pline[ind2 + 1:ind3]
        parg2 = pline[ind3 + 2:ind4]
        try:
            parg2.index(',')
            raise NotImplementedError('Prior with >2 params')
        except ValueError:
            pass
        priors[pname] = {'type': ptype, 'arg1': parg1, 'arg2': parg2}
    return priors


def get_cw_groups(pta):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output. * edited to follow my parameter naming
    """
    ang_pars = ['cos_gwtheta', 'gwphi', 'cos_inc', 'phase0', 'psi']
    mfdh_pars = ['log10_mc', 'log10_fgw', 'log10_dist', 'log10_h']
    freq_pars = ['log10_mc', 'log10_fgw', 'p_dist', 'p_phase']

    groups = []
    for pars in [ang_pars, mfdh_pars, freq_pars]:
        groups.append(group_from_params(pta, pars))

    return groups
