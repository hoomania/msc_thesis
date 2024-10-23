from mcmc import sampling as smpl
from mcmc import validity as vld


## ISING SAMPLE DATA ####
smpl.Sampling(
    lattice_length=10,
    lattice_dim=2,
    lattice_profile='square',
    configs=1,
    temp_low=0.4,
    temp_high=0.7,
    temp_step=0.1,
    samples_per_temp=1,
    beta_inverse=True
).take_sample()

## CHECK DATA ####
length = [10, 20, 30, 40, 50, 60]
config = [8, 0, 8, 8, 5, 9]
i = 0
for ll in length:
    diagram = vld.CheckData(
        f'../data_set/ising_classic/test/configs/L{ll}/data_square_L{ll}_test_config_{config[i]}.csv')
    diagram.magnetization()
    diagram.magnetic_susceptibility()
    diagram.avg_mag_sus()
    diagram.lattice_snapshot()
    # diagram.lattice_brief_view()
    i += 1
