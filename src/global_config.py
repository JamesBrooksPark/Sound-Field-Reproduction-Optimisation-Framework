# ---------------------------------
# === Global Configuration File ===
# ---------------------------------
filter_length = 2**14
NFFT = 2**15

# modelling_delay = 180
modelling_delay = 0

precedence_delay = 0.01

rho = 1
tolerance_admm = 0.005
maximum_iteration = 40
solve_method = 'lu' # 'inverse', 'lu', 'solve'

regularisation_weight = 10**-3

FIR_shaping_weight = 10**-9

time_amp_crossover_Hz = 2000
gamma_vals = [0.4,0.7]

# select the primary source of which the response should be replicated in the playback room
primary_source_index = 0