# partial_sampling_spatially_structured_population
Repository for my lab immersion on spatially structured populations with partial updates.


## Slurm folder
Intended to be run on SCITAS clusters, each script writes a `.json` file with parameters and output data.

- `py well-mixed_simulations.py [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories]`

- `py well-mixed_matrix_computation.py [job_array_nb] [N] [M] [log_s_min] [log_s_max]`

- `py clique_simulations.py [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] [migration_rate] [nb_colonies]`

- `py cycle_simulations.py [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] [migration_rate] [nb_colonies] [alpha]`

- `py star_simulations.py [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] [migration_rate] [nb_colonies] [alpha]`
