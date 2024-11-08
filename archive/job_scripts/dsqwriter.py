"""
This script is made to be called by the dsqwrapper.sh script, but I guess you could use it on its own too.
"""

import sys

maxind = int(sys.argv[1])
ntasks = int(sys.argv[2])

f = open('job_scripts/joblist.txt', 'w')
commands = ['module load miniconda OpenMPI;',
            'conda activate targeted;',
            'export PYTHONPATH=$(pwd):$PATH;',
            f'mpirun -n {ntasks} python scripts/simple_targeted_search.py -t']
command = ' '.join(commands)

for i in range(maxind+1):
    f.write(command + f' {i}\n')
