#!/bin/bash
#SBATCH -J well-mixed_paper_simulations                        # name of the job
#SBATCH -o ./out_well-mixed_paper_simulations%A_%a                     # name of the output file (%A_%a ensures you have a different name for each job: %A is th job number, %a the array number)
#SBATCH -e ./out_well-mixed_paper_simulations%A_%a                     # name of error file (same as output file -> all in one file)
#SBATCH -D /home/prat/                                     # working directory
#SBATCH --array=1-5                                      # this is an array job, where you launch one job per array item (and you can pass different parameters to each job, see below)
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1                                  # number of CPUs for each job
#SBATCH --time 3-00:00:00                                    # maximum allowed time (3 days)
#SBATCH --qos=serial                                       # must add to avoid problems with billing
#SBATCH --mail-type=BEGIN,FAIL,END                         # receive an email if job begins, fail, or ends
#SBATCH --mail-user=noe.prat@epfl.ch                  # change to your email 

module purge
module load  gcc/11.3.0
module load intel/2021.6.0
module load python/3.10.4

source venvs/noe-sim-venv/bin/activate

N=${1}
M=`head -n $SLURM_ARRAY_TASK_ID parameters_WM_paper.txt | tail -n 1 | cut -d' ' -f1`  ## to sweep M
log_s_min=${2}
log_s_max=${3}
nb_trajectories=${4}

python3 /home/prat/well-mixed_simulations.py $SLURM_ARRAY_TASK_ID $N $M $log_s_min $log_s_max $nb_trajectories

## (general) sbatch script_WM_paper.sh [N] [log_s_min] [log_s_max] [nb_trajectories]
## (test) sbatch script_WM_paper.sh 1000 -4 -1 100
## (WM_paper) sbatch script_WM_paper.sh 1000 -4 -1 10000000