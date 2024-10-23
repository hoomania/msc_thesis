import dataset_maker as dsm
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from scipy.interpolate import make_interp_spline

#
mpl.use('TkAgg')
font = {'family': 'cmr10',
        'size': 16}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 150


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#
# def prepare_data(file_path: str, feature_len: int, batch_size: int):
#     return torch.utils.data.DataLoader(
#         dsm.DatasetMaker(
#             file_path,
#             feature_len),
#         batch_size=batch_size,
#         shuffle=False)
#
#
# def plot_kitaev_ladder(
#         model_file_path: str,
#         data_file_path: str,
#         feature_len: int,
#         batch_size: int,
#         fig_dpi: int = 200):
#
#     data = prepare_data(
#         data_file_path,
#         feature_len,
#         batch_size
#     )
#
#     plt.rcParams['figure.dpi'] = fig_dpi
#
#     with open(model_file_path, 'rb') as handle:
#         model = pkl.load(handle)
#
#     dist_list = []
#     for i, (x, y) in enumerate(data):
#         output = model.decoder(model.encoder(x.to(device)))
#         output = output.to('cpu').detach().numpy()
#         output = output.reshape(8)
#
#         dist_list.append(np.sqrt(np.sum((np.array(x) - np.array(output)) ** 2)))
#
#     plt.rcParams['figure.dpi'] = 200
#     plt.imshow(np.array(dist_list).reshape(150, 150), cmap='brg', origin='lower', interpolation='nearest')
#     plt.colorbar()
#     plt.show()
#
#
# def plot_ebh(
#         model_file_path: str,
#         data_file_path: str,
#         feature_len: int,
#         batch_size: int,
#         fig_dpi: int = 200):
#
#     data = prepare_data(
#         data_file_path,
#         feature_len,
#         batch_size
#     )
#
#     matplotlib.use('TkAgg')
#     plt.rcParams['figure.dpi'] = fig_dpi
#
#     with open(model_file_path, 'rb') as handle:
#         model = pkl.load(handle)
#
#     dist_list = []
#     for i, (x, y) in enumerate(data):
#         output = model.decoder(model.encoder(x.to(device)))
#         output = output.to('cpu').detach().numpy()
#         output = output.reshape(8)
#
#         dist_list.append(np.sqrt(np.sum((np.array(x) - np.array(output)) ** 2)))
#
#     plt.rcParams['figure.dpi'] = 200
#     plt.imshow(np.array(dist_list).reshape(61, 41), cmap='brg', origin='lower', interpolation='nearest')
#     plt.colorbar()
#     plt.show()
#
#
# # plot_kitaev_ladder(
# #     model_file_path='model/model_grid_kitaev_ladder.pkl',
# #     data_file_path='../../Dataset/Kitaev_Ladder/lambdas/dataset_lambda_all_labels_hz_sorted.csv',
# #     feature_len=8,
# #     batch_size=1,
# #     fig_dpi=200
# # )
#
# plot_ebh(
#     model_file_path='model/model_grid_ebh.pkl',
#     data_file_path='../../Dataset/Extended_Bose_Hubbard/lambdas/dataset_lambda_all_labels_u_sorted.csv',
#     feature_len=8,
#     batch_size=1,
#     fig_dpi=200
# )

def plot_2d_kitaev_ladder_phase_diagram():
    data_hrz = pd.read_csv(f'../kitaev_ladder/predict/predict_fc_physical_horizontal_D32_all.csv',
                           header=0,
                           index_col=False)

    data_vrt = pd.read_csv(f'../kitaev_ladder/predict/predict_fc_physical_vertical_D32_all.csv',
                           header=0,
                           index_col=False)
    hx_list = data_hrz.sort_values('hx')['hx'].unique()

    data_matrix_hrz = []
    data_matrix_vrt = []
    for i in hx_list:
        data_matrix_hrz.append(data_hrz.query(f'hz == {i}').sort_values('hx')['phase_toric'])
        data_matrix_vrt.append(data_vrt.query(f'hx == {i}').sort_values('hz')['phase_fm'])

    data_matrix_hrz = np.array(data_matrix_hrz)
    data_matrix_vrt = np.transpose(np.array(data_matrix_vrt))
    plt.imshow(data_matrix_hrz, cmap='twilight', origin='lower', interpolation='nearest', alpha=0.5)
    plt.imshow(data_matrix_vrt, cmap='twilight', origin='lower', interpolation='nearest', alpha=0.5)
    plt.colorbar()
    plt.show()


def phase_transition_point(data: pd.DataFrame):
    tc_point_one = data.query(f'phase_toric <= 0.5').iloc[0, :]
    tc_point_two = data.query(f'phase_toric > 0.5').sort_values('phase_toric').iloc[0, :]

    tc_dots = [
        [tc_point_one['x'], tc_point_two['x']],
        [tc_point_one['phase_toric'], tc_point_two['phase_toric']]
    ]
    pl_dots = [
        [tc_point_one['x'], tc_point_two['x']],
        [tc_point_one['phase_fm'], tc_point_two['phase_fm']]
    ]

    fm_slope = (pl_dots[1][1] - pl_dots[1][0]) / (pl_dots[0][1] - pl_dots[0][0])
    pm_slope = (tc_dots[1][1] - tc_dots[1][0]) / (tc_dots[0][1] - tc_dots[0][0])
    fm_b = pl_dots[1][1] - (fm_slope * pl_dots[0][1])
    pm_b = tc_dots[1][1] - (pm_slope * tc_dots[0][1])

    return (pm_b - fm_b) / (fm_slope - pm_slope)


def machine_phases_transition_2d_plot():
    data_hrz = pd.read_csv(f'../kitaev_ladder/predict/predict_fc_physical_horizontal_D32_all.csv',
                           header=0,
                           index_col=False)
    hz_list = data_hrz.sort_values('hz')['hz'].unique()
    hrz_list = []
    for h in hz_list:
        data_line = data_hrz.query(f'hz == {h}')
        data_line = data_line.drop(columns=['hz']).rename(columns={'hx': 'x'}).sort_values('x')
        hrz_list.append(phase_transition_point(data_line))

    data_vrt = pd.read_csv(f'../kitaev_ladder/predict/predict_fc_physical_vertical_D32_all.csv',
                           header=0,
                           index_col=False)
    hx_list = data_vrt.sort_values('hx')['hx'].unique()
    vrt_list = []
    for h in hx_list:
        data_line = data_vrt.query(f'hx == {h}')
        data_line = data_line.drop(columns=['hx']).rename(columns={'hz': 'x'}).sort_values('x')
        vrt_list.append(phase_transition_point(data_line))

    x_list = [i / 100 for i in range(150)]

    plt.figure(figsize=(8, 8))
    plt.plot(hrz_list[:23], x_list[:23], linestyle='--', c='orange', label='Machine')
    plt.plot(x_list[:61], vrt_list[:61], linestyle='--', c='orange')

    grad_data = min_grad()

    y_list = grad_data[0]
    fv = y_list[0]
    indexes = [0]
    index_val = [y_list[0]]
    for i in range(len(y_list)):
        if y_list[i] != fv:
            indexes.append(i / 100)
            index_val.append(y_list[i])
            fv = y_list[i]
    plt.plot(index_val[:10], indexes[:10], c='deepskyblue', label='Simulation')

    y_list = grad_data[1]
    fv = y_list[0]
    indexes = [0]
    index_val = [y_list[0]]
    for i in range(len(y_list)):
        if y_list[i] != fv:
            indexes.append(i / 100)
            index_val.append(y_list[i])
            fv = y_list[i]
    plt.plot(indexes[:8], index_val[:8], c='deepskyblue')
    plt.fill_between(
        [0, 0.35, 0.47, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.636666],
        [0.16, 0.17, 0.18, 0.19, 0.1925, 0.195, 0.1974, 0.20, 0.2020, 0.2041, 0.2060, 0.2080, 0.2100, 0.2150, 0.2200,
         0.223342],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.07, 0.10, 0.13, 0.150, 0.170, 0.190, 0.210, 0.223342],
        color='none', hatch="///", edgecolor="gray", alpha=0.3)
    # plt.grid()
    # # plt.plot(x_list, y_list[1])
    # x_hrz_smth = [0, 0.04, 0.07, 0.10, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.24, 0.26, 0.27]
    # y_hrz_smth = [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67]
    # plt.plot(y_hrz_smth, x_hrz_smth)
    #
    # x_vrt_smth = [0, 0.35, 0.47, 0.52, 0.56, 0.61, 0.63, 0.65, 0.66, 0.68]
    # y_vrt_smth = [0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]
    # plt.plot(x_vrt_smth, y_vrt_smth)

    plt.xlim([0, 0.67])
    plt.ylim([0, 0.67])
    plt.tick_params(direction='in')
    plt.xlabel('$h_z$')
    plt.ylabel('$h_x$')
    plt.annotate('Polarized (x)', xy=(0.4, 0.9))
    plt.annotate('Polarized (z)', xy=(1.1, 0.2))
    plt.annotate('Polarized (x, z)', xy=(1.1, 0.75))
    # plt.annotate('Kitaev (Toric)', xy=(0.2, 0.05))
    plt.legend()
    plt.show()
    # plt.savefig('kitaev_phase_plot_diff_simulation_machine.pdf', format='pdf')


def min_grad():
    data = pd.read_csv(f'../../data_set/kitaev_ladder/physical_profile_D32_all_main.csv',
                       header=0,
                       index_col=False).drop(
        columns=['phy_dim', 'vir_dim', 'unit_cells', 'energy_bond_1', 'energy_bond_2', 'mag_mean_x', 'mag_mean_z'])

    for colName in data.columns[2:]:
        minim = data[colName].min(numeric_only=True)
        maxim = data[colName].max(numeric_only=True)
        data[colName] = (data[colName] - minim) / (maxim - minim)

    data = data.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]]
    # hx_0_toric_phase = data.query('hx == 0 and mag_value <= 0.18')
    const_axis = 'hz'
    second_axis = 'hx'
    target_value = 'energy'
    x_list = []
    y_list = []
    for i in range(150):
        hx_0_toric_phase = data.query(f'{const_axis} == {i / 100}').sort_values(second_axis)
        # print(hx_0_toric_phase)
        y = hx_0_toric_phase[target_value]
        grd_I = np.gradient(np.array(y))
        grd_II = np.gradient(grd_I)
        x_list.append(i / 100)
        y_list.append(np.where(grd_II == np.min(grd_II))[0][0] / 100)
        # print(f'i: {i}, min: {np.where(grd_II == np.min(grd_II))[0]}')

    y_list = np.array(y_list)
    fv = y_list[0]
    indexes = [0]
    index_val = [y_list[0]]
    for i in range(len(y_list)):
        if y_list[i] != fv:
            indexes.append(i / 100)
            index_val.append(y_list[i])
            fv = y_list[i]

    # plt.figure(figsize=(8, 8))
    # plt.plot(index_val, indexes, label='Min of $\\frac{d^2}{dh^2_x}\langle E \\rangle$')

    y_hrz = np.array(y_list)
    const_axis = 'hx'
    second_axis = 'hz'
    target_value = 'energy'
    x_list = []
    y_list = []
    for i in range(150):
        hx_0_toric_phase = data.query(f'{const_axis} == {i / 100}').sort_values(second_axis)
        # print(hx_0_toric_phase)
        y = hx_0_toric_phase[target_value]
        grd_I = np.gradient(np.array(y))
        grd_II = np.gradient(grd_I)
        x_list.append(i / 100)
        y_list.append(np.where(grd_II == np.min(grd_II))[0][0] / 100)
        # print(f'i: {i}, min: {np.where(grd_II == np.min(grd_II))[0]}')

    y_list = np.array(y_list)
    fv = y_list[0]
    indexes = [0]
    index_val = [y_list[0]]
    for i in range(len(y_list)):
        if y_list[i] != fv:
            indexes.append(i / 100)
            index_val.append(y_list[i])
            fv = y_list[i]

    # plt.plot(indexes, index_val, label='Min of $\\frac{d^2}{dh^2_z}\langle E \\rangle$')
    #
    # plt.xlim([0, 1.49])
    # plt.ylim([0, 1.49])
    # plt.tick_params(direction='in')
    # plt.xlabel('$h_z$')
    # plt.ylabel('$h_x$')
    # plt.annotate('Polarized (x)', xy=(0.4, 0.9))
    # plt.annotate('Polarized (z)', xy=(1.1, 0.2))
    # plt.annotate('Polarized (x, z)', xy=(1.1, 0.75))
    # plt.annotate('Kitaev (Toric)', xy=(0.2, 0.05))
    # plt.legend()
    # # plt.show()
    # # plt.savefig('kitaev_phase_plot_energy_second_derivative.pdf', format='pdf')

    y_vrt = np.array(y_list)

    return y_hrz, y_vrt


# plot_2d_kitaev_ladder_phase_diagram()
machine_phases_transition_2d_plot()
# min_grad()
