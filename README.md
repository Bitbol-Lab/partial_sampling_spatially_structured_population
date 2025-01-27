# partial_sampling_spatially_structured_population
Repository for my lab immersion on spatially structured populations with partial updates.


## Run simulations / computations

### slurm-main.py

`py slurm-main.py [type] [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

The arguments in parentheses depend on the chosen type:

- For clique graphs: `py slurm-main.py clique [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes)`

- For cycle graphs: `py slurm-main.py cycle [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha)`

- For star graphs: `py slurm-main.py star [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

- For line graphs: `py slurm-main.py line [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

> You can choose to have the averaged fixation probability over the possible initial nodes if `initial node = 'avg'`. In that case, it is not possible to store the fixation/extinction times.

- For well-mixed simulations: `py slurm-main.py wm_sim [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories]`

- For well-mixed matrix computations: `py slurm-main.py wm_mat [job_array_nb] [N] [M] [log_s_min] [log_s_max]`

### run_single.py

`py run_single.py [type] [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

The arguments in parentheses depend on the chosen type:

- For clique graphs: `py run_single.py clique [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes)`

- For cycle graphs: `py run_single.py cycle [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes) (alpha)`

- For star graphs: `py run_single.py star [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

- For line graphs: `py run_single.py line [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

- For well-mixed simulations: `py run_single.py wm_sim [job_array_nb] [N] [M] [s] [nb_trajectories]`

- For well-mixed matrix computations: `py run_single.py wm_mat [job_array_nb] [N] [M] [s]`


