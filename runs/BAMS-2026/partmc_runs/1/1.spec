run_type particle
output_prefix out/1
n_repeat 1
n_part 10000
restart no
do_select_weighting yes
weight_type power_source
weighting_exponent 0

t_max 3600
del_t 60
t_output 60
t_progress 60

do_camp_chem no

do_tchem no
gas_data gas_data.dat
gas_init gas_init.dat

aerosol_data aero_data.dat
do_fractal no
aerosol_init aero_init_dist.dat

temp_profile temp.dat
pressure_profile pres.dat
height_profile height.dat
gas_emissions gas_emit.dat
gas_background gas_back.dat
aero_emissions aero_emit.dat
aero_background aero_back.dat
loss_function none

rel_humidity 0.0
latitude 0
longitude 0
altitude 3000.0
start_time 21600
start_day 200

do_coagulation yes
coag_kernel brown
do_condensation no
do_mosaic yes
do_optical no
do_nucleation no

rand_init 42
allow_doubling yes
allow_halving yes
record_removals yes
do_parallel no