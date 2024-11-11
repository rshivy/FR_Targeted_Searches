### The below is out of date, mostly disregard

The main script here is `scripts/simple_targeted_search.py`, which is a simplified version of Bjorn's script. I commented everything I knew, 
although there are a few things I'm still confused about.

## Setting up

You can clone this whole repository into a location of your choice on Grace and it should work. You will need a conda environment with
- enterprise_extensions
- QuickCW

You can get enterprise_extensions with `conda install enterprise_extensions`, but to install 
QuickCW follow the instructions [here](https://github.com/nanograv/QuickCW/blob/main/docs/how_to_run_QuickCW.md).


## Using simple_targeted_search scripts

- run all commands from inside FR_Targeted_Searches, but not in any subdirectories
- edit `job_scripts/simple_targeted_search.sh` to activate your conda environment on line 11
- edit `scripts/simple_targeted_search.py` to your name on line 12
- Run the command `sbatch job_scripts/simple_targeted_search.sh` from the command line.

You can monitor the progress of the sampler by checking the log files in `logs/simple.txt`, and if anything goes wrong you can check error messages in `logs/error/simple.txt`.
By default the output will all be in `data/chains/ng15_v1p1/ZTF18abxxohm`.
