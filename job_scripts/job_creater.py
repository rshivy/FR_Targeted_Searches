"""
Work in progress allowing targeted searches with flexible combination of targets
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-o', '--out', action='store', type=str, dest='output_suffix')
parser.add_argument('-t', '--target', action='store', dest='target_index')
parser.add_argument('-m', '--mass-prior', action='store', dest='mass_prior',
                    choices=['upper-limit', 'detection'], required=True)
parser.add_argument('-f', '--frequency-prior', action='store', dest='frequency_prior',
                    choices=['constant', 'narrow', 'full'], default='narrow')
parser.add_argument('-n', '--num-pt-tasks', action='store', type=int, dest='num_pt_tasks')

args = parser.parse_args()

try:
    target = int(args.target_index)
    multi = False
    job_prefix = f't{int(target)}'
except ValueError:
    targets = args.target_index.split(',')
    multi = True
    job_prefix = 't+'

if args.mass_prior == 'upper-limit':
    mass_prior = 'ul'
else:
    mass_prior = 'det'

if args.frequency_prior == 'narrow':
    frequency_prior = 'narrowfgw'
elif args.frequency_prior == 'full':
    frequency_prior = 'varyfgw'
else:
    frequency_prior = None

if frequency_prior:
    job_name = f'{job_prefix}-{mass_prior}-{frequency_prior}'
else:
    job_name = f'{job_prefix}-{mass_prior}'

script = '#!/usr/bin/env bash\n'


script += '#SBATCH --job-name=t1-21-det\n'
script += f'#SBATCH --job-name={job_name}\n'
script += '#SBATCH --time=3-00:00:00'
script += '#SBATCH --partition=pi_mingarelli'
