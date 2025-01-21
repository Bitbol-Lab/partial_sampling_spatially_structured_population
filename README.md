# partial_sampling_spatially_structured_population
Repository for my lab immersion on spatially structured populations with partial updates.


## Slurm folder
Intended to be run on SCITAS clusters, each script writes a `.json` file with parameters and output data in `./results`.

- `py well-mixed_simulations.py [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories]`

- `py well-mixed_matrix_computation.py [job_array_nb] [N] [M] [log_s_min] [log_s_max]`

- `py clique_simulations.py [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] [migration_rate] [nb_colonies]`

- `py cycle_simulations.py [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] [migration_rate] [nb_colonies] [alpha]`

- `py star_simulations.py [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] [migration_rate] [nb_colonies] [alpha]`


Should be replaced by running: 

`py slurm-main [type] [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

The arguments in parentheses depend on the chosen type:

- For clique graphs: `py slurm-main clique [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes)`

- For cycle graphs: `py slurm-main cycle [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha)`

- For star graphs: `py slurm-main star [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

- For well-mixed simulations: `py slurm-main wm_sim [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories]`

- For well-mixed matrix computations: `py slurm-main wm_mat [job_array_nb] [N] [M] [log_s_min] [log_s_max]`


