import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
font = {'family': 'cmr10',
        'size': 12}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 150


def lambda_supervised_dataset():
    data = pd.read_csv(f'../../data_set/kitaev_ladder/lambda_full_profile.csv',
                       header=0,
                       index_col=False)
    hx_001_toric_phase = data.query('hx == 0.01 and mag <= 0.11').drop(columns=['round_mag', 'class'])
    hx_001_toric_phase.insert(hx_001_toric_phase.shape[1], 'phase', [0 for _ in range(hx_001_toric_phase.shape[0])],
                              True)
    hx_001_fm_phase = data.query('hx == 0.01 and mag >= 0.97').drop(columns=['round_mag', 'class'])
    hx_001_fm_phase.insert(hx_001_fm_phase.shape[1], 'phase', [1 for _ in range(hx_001_fm_phase.shape[0])], True)
    hz_001_toric_phase = data.query('hz == 0.01 and mag <= 0.11').drop(columns=['round_mag', 'class'])
    hz_001_toric_phase.insert(hz_001_toric_phase.shape[1], 'phase', [0 for _ in range(hz_001_toric_phase.shape[0])],
                              True)

    train = pd.concat([hx_001_toric_phase, hz_001_toric_phase, hx_001_fm_phase])
    train.to_csv(path_or_buf='../../data_set/kitaev_ladder/fc/lambda_D32_train.csv', index=False)

    hx_000 = data.query('hx == 0.0').drop(columns=['round_mag', 'class'])
    hz_000 = data.query('hz == 0.0').drop(columns=['round_mag', 'class'])

    test = pd.concat([hx_000, hz_000])
    test.to_csv(path_or_buf='../../data_set/kitaev_ladder/fc/lambda_D32_test.csv', index=False)


def physical_properties_supervised_dataset():
    data = pd.read_csv(f'../../data_set/kitaev_ladder/physical_profile_D32_all_main.csv',
                       header=0,
                       index_col=False).drop(
        columns=['phy_dim', 'vir_dim', 'unit_cells', 'energy_bond_1', 'energy_bond_2', 'mag_mean_x', 'mag_mean_z'])

    for colName in data.columns[2:]:
        minim = data[colName].min(numeric_only=True)
        maxim = data[colName].max(numeric_only=True)
        data[colName] = (data[colName] - minim) / (maxim - minim)

    data = data.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]]

    # three phase ->
    # toric_phase_1 = data.query(f'hz == 0 and hx < 0.27')
    # toric_phase_2 = data.query(f'hz == 0.01 and hx < 0.27')
    # toric_phase_3 = data.query(f'hx == 0 and hz < 0.08')
    # toric_phase_4 = data.query(f'hx == 0.01 and hz < 0.08')
    # toric_phase = pd.concat([toric_phase_1, toric_phase_2, toric_phase_3, toric_phase_4])
    # toric_phase.insert(toric_phase.shape[1], 'phase', [0 for _ in range(toric_phase.shape[0])], True)
    # print(np.array(toric_phase['mag_value']).shape)
    #
    # fm_up_1 = data.query(f'hz == 0 and hx > 1.14')
    # fm_up_2 = data.query(f'hz == 0.01 and hx > 1.14')
    # fm_up = pd.concat([fm_up_1, fm_up_2])
    # fm_up.insert(fm_up.shape[1], 'phase', [1 for _ in range(fm_up.shape[0])], True)
    # print(np.array(fm_up['mag_value']).shape)
    #
    # fm_down_1 = data.query(f'hx == 0 and hz > 1.14')
    # fm_down_2 = data.query(f'hx == 0.01 and hz > 1.14')
    # fm_down = pd.concat([fm_down_1, fm_down_2])
    # fm_down.insert(fm_down.shape[1], 'phase', [2 for _ in range(fm_down.shape[0])], True)
    # print(np.array(fm_down['mag_value']).shape)
    #
    # train = pd.concat([toric_phase, fm_up, fm_down])
    # train.to_csv(path_or_buf='../../data_set/kitaev_ladder/fc/physical_three_phases_D32_train.csv', index=False)

    # hx -> 0
    # toric_phase = data.query(f'hz == 0 and hx < 0.27')
    # toric_phase.insert(toric_phase.shape[1], 'phase', [0 for _ in range(toric_phase.shape[0])], True)
    # print(np.array(toric_phase['mag_value']).shape)
    #
    # fm_up = data.query(f'hz == 0 and hx > 1.22')
    # fm_up.insert(fm_up.shape[1], 'phase', [1 for _ in range(fm_up.shape[0])], True)
    # print(np.array(fm_up['mag_value']).shape)
    #
    # train = pd.concat([toric_phase, fm_up])
    # train.to_csv(path_or_buf='../../data_set/kitaev_ladder/fc/physical_horizontal_D32_train.csv', index=False)

    # hz -> 0
    toric_phase_3 = data.query(f'hx == 0 and hz < 0.08')
    toric_phase_4 = data.query(f'hx == 0.01 and hz < 0.08')
    toric_phase = pd.concat([toric_phase_3, toric_phase_4])
    toric_phase.insert(toric_phase.shape[1], 'phase', [0 for _ in range(toric_phase.shape[0])], True)
    print(np.array(toric_phase['mag_value']).shape)

    fm_down_1 = data.query(f'hx == 0 and hz > 1.41')
    fm_down_2 = data.query(f'hx == 0.01 and hz > 1.41')
    fm_down = pd.concat([fm_down_1, fm_down_2])
    fm_down.insert(fm_down.shape[1], 'phase', [1 for _ in range(fm_down.shape[0])], True)
    print(np.array(fm_down['mag_value']).shape)

    train = pd.concat([toric_phase, fm_down])
    train.to_csv(path_or_buf='../../data_set/kitaev_ladder/fc/physical_vertical_D32_train.csv', index=False)


def lambda_dataset():
    data = pd.read_csv(f'../../data_set/kitaev_ladder/lambda_D32_all_main.csv',
                       header=0,
                       index_col=False).drop(columns=['round_mag', 'class'])

    all = data.drop(columns=['mag'])
    all.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/lambda_D32_all.csv',
               index=False)

    tc_phase = data.query('hz <= 0.09 and hx >= 1.4 and mag >= 0.94').drop(columns=['mag'])
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/lambda_D32_plzd_x_phase.csv',
                    index=False)

    tc_phase = data.query('hz >= 1.4 and hx <= 0.09 and mag >= 0.95').drop(columns=['mag'])
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/lambda_D32_plzd_z_phase.csv',
                    index=False)

    tc_phase = data.query('hz <= 0.12 and hx <= 0.12 and mag <= 0.09').drop(columns=['mag'])
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/lambda_D32_toric_phase.csv',
                    index=False)


def physical_profile_normalized_dataset():
    data = pd.read_csv(f'../../data_set/kitaev_ladder/physical_profile_D32_all_main.csv',
                       header=0,
                       index_col=False).drop(
        columns=['phy_dim', 'vir_dim', 'unit_cells', 'energy_bond_1', 'energy_bond_2', 'mag_mean_x', 'mag_mean_z'])

    for colName in data.columns[2:]:
        minim = data[colName].min(numeric_only=True)
        maxim = data[colName].max(numeric_only=True)
        data[colName] = (data[colName] - minim) / (maxim - minim)

    data = data.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]]
    data.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_all.csv', index=False)

    tc_phase = data.query('hz <= 0.09 and hx >= 1.4 and mag_value >= 0.95')
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_plzd_x_phase.csv',
                    index=False)

    tc_phase = data.query('hz >= 1.4 and hx <= 0.09 and mag_value >= 0.95')
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_plzd_z_phase.csv',
                    index=False)

    tc_phase = data.query('hz <= 0.09 and hx <= 0.09 and mag_value <= 0.05')
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_toric_phase.csv',
                    index=False)


def lambda_normalized_dataset():
    data = pd.read_csv(f'../../data_set/kitaev_ladder/lambda_D32_all_main.csv',
                       header=0,
                       index_col=False).drop(columns=['round_mag', 'class'])

    for colName in data.columns[:-3]:
        minim = data[colName].min(numeric_only=True)
        maxim = data[colName].max(numeric_only=True)
        print(colName, minim, maxim)
        # print(data[colName])
        data[colName] = (data[colName] - minim) / (maxim - minim)
        print(data[colName])

    #
    # data = data.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]]
    data.to_csv(path_or_buf='../../data_set/kitaev_ladder/lambda_normalized_D32_all_main.csv', index=False)

    data.drop(columns=['mag']).to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/lambda_normalized_D32_all.csv',
                                      index=False)

    tc_phase = data.query('hz <= 0.09 and hx >= 1.4 and mag >= 0.95').drop(columns=['mag'])
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/lambda_normalized_D32_plzd_x_phase.csv',
                    index=False)

    tc_phase = data.query('hz >= 1.4 and hx <= 0.09 and mag >= 0.95').drop(columns=['mag'])
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/lambda_normalized_D32_plzd_z_phase.csv',
                    index=False)

    tc_phase = data.query('hz <= 0.09 and hx <= 0.09 and mag <= 0.05').drop(columns=['mag'])
    tc_phase.to_csv(path_or_buf='../../data_set/kitaev_ladder/ae/lambda_normalized_D32_toric_phase.csv',
                    index=False)


def data_boxplot():
    # data = pd.read_csv(
    #     f'../../data_set/kitaev_ladder/ae/lambda_D32_all_main.csv',
    #     header=0,
    #     index_col=False).drop(columns=['hx', 'hz', 'mag', 'round_mag', 'class'])

    data = pd.read_csv(
        f'../../data_set/kitaev_ladder/ae/physical_profile_D32_all.csv',
        header=0,
        index_col=False).drop(columns=['energy', 'hx', 'hz'])

    # data = pd.read_csv(
    #     f'../../data_set/kitaev_ladder/ae/pca_D32_all.csv',
    #     header=0,
    #     index_col=False).drop(columns=['hx', 'hz'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, 0, '')
    ax.set(
        title='Boxplot of feature vector elements',
        xlabel='Element (columns)',
        ylabel='')

    plt.tick_params(direction='in')
    plt.show()


def energy_plot():
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
    const_axis = 'hx'
    second_axis = 'hz'
    target_value = 'energy'
    hx_0_toric_phase = data.query(f'{const_axis} == 0.55').sort_values(second_axis)
    # print(hx_0_toric_phase)
    y = hx_0_toric_phase[target_value]
    grd_I = np.gradient(np.array(y))
    grd_II = np.gradient(grd_I)

    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.scatter(hx_0_toric_phase[second_axis], y, s=0.2)
    # ax.plot(hx_0_toric_phase[second_axis], y)
    # ax.plot(hx_0_toric_phase[second_axis], grd_I*100)
    ax.scatter(hx_0_toric_phase[second_axis], grd_II*1000)
    ax.set(
        # title='Probability of ' + 'FM' if phase_name == 'phase_f' else 'PM ' + f'phase (L={self.lattice_length})',
        # title=f'D=32',
        xlabel='$h_z$',
        # ylabel='$\langle E \\rangle$',
        # ylabel='$\\frac{d}{dh_z}\langle E \\rangle \quad (\\times 10^{-2})$',
        ylabel='$\\frac{d^2}{dh^2_z}\langle E \\rangle \quad (\\times 10^{-3})$',
        xticks=(np.arange(150) / 100),
        xticklabels=hx_0_toric_phase[second_axis])

    plt.axvline(
        x=0.55,
        color='gray',
        linestyle='dotted',
        label='Critical Point')
    plt.tick_params(direction='in')
    plt.locator_params(axis='x', nbins=10)
    # plt.tick_params(left=False, labelleft=False)
    plt.legend()
    plt.show()
    # plt.savefig(f'grd_II_energy_plot_hx_0.pdf', format='pdf')

    # plt.scatter(hx_0_toric_phase[second_axis], y)
    # plt.show()
    # plt.scatter(hx_0_toric_phase[second_axis], grd_I)
    # plt.show()
    # plt.scatter(hx_0_toric_phase[second_axis], grd_II)
    # plt.show()


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
        hx_0_toric_phase = data.query(f'{const_axis} == {i/100}').sort_values(second_axis)
        # print(hx_0_toric_phase)
        y = hx_0_toric_phase[target_value]
        grd_I = np.gradient(np.array(y))
        grd_II = np.gradient(grd_I)
        x_list.append(i/100)
        y_list.append(np.where(grd_II == np.min(grd_II))[0][0]/100)
        # print(f'i: {i}, min: {np.where(grd_II == np.min(grd_II))[0]}')

    plt.plot(np.array(y_list), np.array(x_list))

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
        y_list.append(np.where(grd_II == np.min(grd_II))[0][0]/100)
        # print(f'i: {i}, min: {np.where(grd_II == np.min(grd_II))[0]}')
    plt.plot(np.array(x_list), np.array(y_list))
    plt.xlim([0.0, 1.49])
    plt.ylim([0.0, 1.49])
    plt.show()




# lambda_normalized_dataset()
# lambda_dataset()
# physical_profile_normalized_dataset()
# physical_profile_dataset()
# data_boxplot()
# physical_properties_supervised_dataset()
# energy_plot()
min_grad()
