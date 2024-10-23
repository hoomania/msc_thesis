import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
font = {'family': 'cmr10',
        'size': 14}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 300


class Diagram:
    def __init__(self):
        pass

    def matrix_row_nrm(self, matrix: np.ndarray):
        result = []
        for i in range(matrix.shape[0]):
            mn = matrix[i:].min()
            mx = matrix[i:].max()
            result.append((matrix[i, :] - mn) / (mx - mn))

        return np.array(result)

    def predict_phase_profile(self, phase_name: str, data_type: str) -> dict:
        if data_type == 'physical':
            if phase_name == 'mi':
                data = pd.read_csv(f'./predict/predict_ae_D16_physical_mi.csv')
            elif phase_name == 'sf':
                data = pd.read_csv(f'./predict/predict_ae_D16_physical_sf.csv')
            elif phase_name == 'hi':
                data = pd.read_csv(f'./predict/predict_ae_D16_physical_hi.csv')
            else:
                data = pd.read_csv(f'./predict/predict_ae_D16_physical_dw.csv')
        else:
            if phase_name == 'mi':
                data = pd.read_csv(f'./predict/predict_ae_D16_lambda_mi.csv')
            elif phase_name == 'sf':
                data = pd.read_csv(f'./predict/predict_ae_D16_lambda_sf.csv')
            elif phase_name == 'hi':
                data = pd.read_csv(f'./predict/predict_ae_D16_lambda_hi.csv')
            else:
                data = pd.read_csv(f'./predict/predict_ae_D16_lambda_dw.csv')

        data_matrix = []
        for step in data.sort_values('u')['u'].unique():
            data_matrix.append(data.query(f'u == {step}').sort_values('v')['loss'])

        data_matrix = np.array(data_matrix)

        minim = data_matrix.min()
        maxim = data_matrix.max()
        data_matrix_nrm = (data_matrix - minim) / (maxim - minim)

        grd_I_x = np.gradient(data_matrix, axis=1)
        grd_I_y = np.gradient(data_matrix, axis=0)
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        grd_I_x_nrm = self.matrix_row_nrm(grd_I_x)
        grd_I_y_nrm = self.matrix_row_nrm(grd_I_y)
        grd_II_x_nrm = self.matrix_row_nrm(grd_II_x)
        grd_II_y_nrm = self.matrix_row_nrm(grd_II_y)

        return {
            'data_matrix': data_matrix,
            'data_matrix_nrm': data_matrix_nrm,
            'grd_i_x': grd_I_x,
            'grd_i_y': grd_I_y,
            'grd_ii_x': grd_II_x,
            'grd_ii_y': grd_II_y,
            'grd_i_x_nrm': grd_I_x_nrm,
            'grd_i_y_nrm': grd_I_y_nrm,
            'grd_ii_x_nrm': grd_II_x_nrm,
            'grd_ii_y_nrm': grd_II_y_nrm
        }

    def ae_phase_diagram(self, phase_name: str, data_type: str):

        matrix = self.predict_phase_profile(phase_name, data_type)

        # Depict illustration
        plt.figure(figsize=(12, 8))
        plt.imshow(
            matrix['data_matrix_nrm'],
            cmap='viridis',
            origin='lower',
            interpolation='nearest',
            aspect='auto')
        plt.colorbar(label='loss', fraction=0.046, pad=0.04)

        plt.xlabel('$V$')
        plt.ylabel('$U$')
        plt.tick_params(direction='in')
        plt.xticks([i for i in np.arange(0, 40, 5)], labels=[i / 10 for i in np.arange(0, 40, 5)])
        plt.yticks([i for i in np.arange(0, 60, 5)], labels=[i / 10 for i in np.arange(0, 60, 5)])

        plt.annotate('MI', xy=(10, 50), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('SF', xy=(5, 20), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('HI', xy=(17, 20), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('DW', xy=(30, 20), color='red',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))

        plt.savefig(f'ea_phase_diagram_{phase_name}.pdf', format='pdf')
        # plt.show()

    def ae_stream_diagram(self, phase_name: str, data_type: str):
        matrix = self.predict_phase_profile(phase_name, data_type)

        # 1D arrays
        feature_x = np.arange(0, 41, 1)
        feature_y = np.arange(0, 61, 1)

        # Creating 2-D grid of features
        x, y = np.meshgrid(feature_x, feature_y)

        # Depict illustration
        plt.figure(figsize=(12, 8))
        ax = plt.axes()
        ax.set_facecolor("#e4e4e4")

        # plt.quiver(x, y, matrix['grd_ii_x'], matrix['grd_ii_y'])
        plt.streamplot(x, y, matrix['grd_ii_x'], matrix['grd_ii_y'],
                       density=3.2,
                       linewidth=matrix['data_matrix_nrm'] + 0.50,
                       color=matrix['data_matrix_nrm'],
                       cmap='viridis',
                       integration_direction='both',
                       broken_streamlines=True)
        plt.colorbar(label='loss', fraction=0.046, pad=0.04)

        plt.xlabel('$v$')
        plt.ylabel('$u$')
        plt.tick_params(direction='in')
        plt.xticks([i for i in np.arange(0, 40, 5)], labels=[i / 10 for i in np.arange(0, 40, 5)])
        plt.yticks([i for i in np.arange(0, 60, 5)], labels=[i / 10 for i in np.arange(0, 60, 5)])

        plt.annotate('mi', xy=(10, 50), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('sf', xy=(5, 20), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('hi', xy=(17, 20), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('dw', xy=(30, 20), color='red',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))

        plt.savefig(f'ea_stream_diagram_{phase_name}.pdf', format='pdf')
        plt.show()

    def four_phase_line_plot(self, phase_name: str, data_type: str):
        matrix = self.predict_phase_profile(phase_name, data_type)

        plt.figure(figsize=(12, 8))
        plt.plot([i for i in range(matrix['data_matrix_nrm'].shape[1])], matrix['data_matrix_nrm'][40, :], '-o',
                 color='#1f77b4', mfc='#ff7f0e')

        pt_lines = {
            'mi': [18, 24, 29],
            'sf': [18, 24, 29],
            'hi': [18, 24, 29],
            'dw': [18, 24, 29]
        }

        for pt in pt_lines[phase_name]:
            plt.axvline(
                x=pt,
                color='gray',
                linestyle='--',
                label='1.8')

        plt.annotate('MI', xy=(8, 1), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('SF', xy=(21, 1), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('HI', xy=(26, 1), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('DW', xy=(34, 1), color='red',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))

        plt.title('U=4.0')
        plt.ylim([-0.1, 1.1])
        plt.xlabel('$V$')
        plt.ylabel('$Loss$')
        plt.tick_params(direction='in')
        plt.xticks([i for i in np.arange(0, 41, 5)], labels=[i / 10 for i in np.arange(0, 41, 5)])
        plt.savefig(f'ae_phase_line_{phase_name}.pdf', format='pdf')
        plt.show()

    def mu_secret(self):
        data = pd.read_csv('../../data_set/extended_bose_hubbard/physical_D16_all_mu.csv')

        u_list = data.sort_values('u')['u'].unique()
        mu_list = []

        for u in u_list:
            mu_list.append(data.query(f'u == {u}').sort_values('v')['b-energy-i'])

        plt.figure(figsize=(12, 8))
        plt.imshow(
            np.array(mu_list),
            cmap='viridis',
            origin='lower',
            interpolation='nearest',
            aspect='auto')
        plt.colorbar(label='loss', fraction=0.046, pad=0.04)
        plt.show()

        data_matrix = np.array(mu_list)
        minim = data_matrix.min()
        maxim = data_matrix.max()
        data_matrix_nrm = (data_matrix - minim) / (maxim - minim)

        grd_I_x = np.gradient(data_matrix, axis=1)
        grd_I_y = np.gradient(data_matrix, axis=0)
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        # 1D arrays
        feature_x = np.arange(0, 41, 1)
        feature_y = np.arange(0, 61, 1)

        # Creating 2-D grid of features
        x, y = np.meshgrid(feature_x, feature_y)

        # Depict illustration
        plt.figure(figsize=(12, 8))
        ax = plt.axes()
        ax.set_facecolor("#e4e4e4")

        # plt.quiver(x, y, matrix['grd_ii_x'], matrix['grd_ii_y'])
        plt.streamplot(x, y, grd_II_x, grd_II_y,
                       density=3.2,
                       # linewidth=matrix['data_matrix_nrm'] + 0.50,
                       # color=matrix['data_matrix_nrm'],
                       cmap='viridis',
                       integration_direction='both',
                       broken_streamlines=True)
        plt.colorbar(label='loss', fraction=0.046, pad=0.04)

        plt.annotate('mi', xy=(10, 50), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('sf', xy=(5, 20), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('hi', xy=(17, 20), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        plt.annotate('dw', xy=(30, 20), color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))

        plt.show()
