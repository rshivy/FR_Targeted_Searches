"""
This script is made to be called by the dsqwrapper.sh script, but I guess you could use it on its own too.
"""

import sys

if len(sys.argv) > 1:
    maxind = int(sys.argv[1])
else:
    maxind = 111

f = open('joblist.txt', 'w')
commands = ['module load miniconda OpenMPI;',
            'conda activate targeted;',
            'export PYTHONPATH=$(pwd):$PATH;',
            'mpirun -n 4 python scripts/simple_targeted_search.py -t']
command = ' '.join(commands)

for i in range(1, maxind+1):
    f.write(command + f' {i}\n')
