import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import diagrams as dgrm

mpl.use('TkAgg')
font = {'family': 'cmr10',
        'size': 12}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 300


class Diagram:
    def __init__(self,
                 virtual_dimension: int):
        self.virtual_dimension = virtual_dimension

    def physics_profile(self, target_col: str = 'energy') -> dict:
        data = pd.read_csv(f'../../data_set/kitaev_ladder/physical_profile_D32_all_main.csv')
        hz_list = data.sort_values('hz')['hz'].unique()

        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')[target_col])

        data_matrix = np.array(data_matrix)

        minim = data_matrix.min()
        maxim = data_matrix.max()
        data_matrix_nrm = (data_matrix - minim) / (maxim - minim)

        grd_I_mtx = np.gradient(data_matrix)
        grd_I_x = grd_I_mtx[1]
        grd_I_y = grd_I_mtx[0]
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

    def matrix_row_nrm(self, matrix: np.ndarray):
        result = []
        for i in range(matrix.shape[0]):
            mn = matrix[i:].min()
            mx = matrix[i:].max()
            result.append((matrix[i, :] - mn) / (mx - mn))

        return np.array(result)

    def phase_transition_border_energy_data(self):
        physics_profile = self.physics_profile('energy')
        grd_ii_x = physics_profile['grd_ii_x']
        grd_ii_y = physics_profile['grd_ii_y']

        y_list = []
        for i in range(150):
            y_list.append(np.where(grd_ii_x == np.min(grd_ii_x[i, :]))[1][0] / 100)

        y_list = np.array(y_list)
        fv = y_list[0]
        x_val = [0]
        y_val = [y_list[0]]
        for i in range(len(y_list)):
            if y_list[i] != fv:
                x_val.append(i / 100)
                y_val.append(y_list[i])
                fv = y_list[i]

        horizontal_cut = {
            'x': y_val,
            'y': x_val,
        }

        y_list = []
        for i in range(150):
            y_list.append(np.where(grd_ii_y == np.min(grd_ii_y[:, i]))[0][0] / 100)

        y_list = np.array(y_list)
        fv = y_list[0]
        x_val = [0]
        y_val = [y_list[0]]
        for i in range(len(y_list)):
            if y_list[i] != fv:
                x_val.append(i / 100)
                y_val.append(y_list[i])
                fv = y_list[i]

        vertical_cut = {
            'x': x_val,
            'y': y_val,
        }

        return {
            'horizontal_cut': horizontal_cut,
            'vertical_cut': vertical_cut,
        }

    def phase_transition_border_energy_plot(self):
        data = self.phase_transition_border_energy_data()

        plt.figure(figsize=(8, 8))
        plt.plot(data['vertical_cut']['x'], data['vertical_cut']['y'])
        plt.plot(data['horizontal_cut']['x'], data['horizontal_cut']['y'])
        plt.xlim([0, 1.49])
        plt.ylim([0, 1.49])
        plt.tick_params(direction='in')
        plt.xlabel('$h_z$')
        plt.ylabel('$h_x$')
        plt.annotate('Polarized (x)', xy=(0.4, 0.9))
        plt.annotate('Polarized (z)', xy=(1.1, 0.2))
        plt.annotate('Polarized (x, z)', xy=(1.1, 0.75))
        plt.annotate('Kitaev (Toric)', xy=(0.2, 0.05))
        # plt.legend()
        plt.show()

    def predict_phase_profile(self, phase_name: str, data_type: str) -> dict:
        if data_type == 'physical':
            if phase_name == 'toric':
                data = pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_toric.csv')
            elif phase_name == 'plzx':
                data = pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_plzd_x.csv')
            else:
                data = pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_plzd_z.csv')
        else:
            if phase_name == 'toric':
                data = pd.read_csv(f'./predict/predict_ae_D32_lambda_toric.csv')
            elif phase_name == 'plzx':
                data = pd.read_csv(f'./predict/predict_ae_D32_lambda_plzd_x.csv')
            else:
                data = pd.read_csv(f'./predict/predict_ae_D32_lambda_plzd_z.csv')


        data_matrix = []
        for step in data.sort_values('hz')['hz'].unique():
            data_matrix.append(data.query(f'hz == {step}').sort_values('hx')['loss'])

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

    def phase_transition_border_predict_data(self, phase_name: str):
        profile = self.predict_phase_profile(phase_name)
        grd_i_x = profile['grd_i_x_nrm']

        y_list = []
        for i in range(150):
            y_list.append(np.where(grd_i_x[i, :] == np.max(grd_i_x[i, :]))[0][0] / 100)

        y_list = np.array(y_list)
        fv = y_list[0]
        x_val = [0]
        y_val = [y_list[0]]
        for i in range(len(y_list)):
            if y_list[i] != fv:
                x_val.append(i / 100)
                y_val.append(y_list[i])
                fv = y_list[i]

        horizontal_cut = {
            'x': y_val,
            'y': x_val,
        }

        grd_i_y = profile['grd_i_y_nrm']
        y_list = []
        for i in range(150):
            y_list.append(np.where(grd_i_y[:, i] == np.max(grd_i_y[:, i]))[0][0] / 100)

        y_list = np.array(y_list)
        fv = y_list[0]
        x_val = [0]
        y_val = [y_list[0]]
        for i in range(len(y_list)):
            if y_list[i] != fv:
                x_val.append(i / 100)
                y_val.append(y_list[i])
                fv = y_list[i]

        vertical_cut = {
            'x': x_val,
            'y': y_val,
        }

        return {
            'horizontal_cut': horizontal_cut,
            'vertical_cut': vertical_cut,
        }

    def phase_transition_border_predict_plot(self, phase_name: str):
        data = self.phase_transition_border_predict_data(phase_name)

        plt.figure(figsize=(8, 8))
        plt.plot(data['vertical_cut']['x'], data['vertical_cut']['y'])
        plt.plot(data['horizontal_cut']['x'], data['horizontal_cut']['y'])
        plt.xlim([0, 1.49])
        plt.ylim([0, 1.49])
        plt.tick_params(direction='in')
        plt.xlabel('$h_z$')
        plt.ylabel('$h_x$')
        plt.annotate('Polarized (x)', xy=(0.4, 0.9))
        plt.annotate('Polarized (z)', xy=(1.1, 0.2))
        plt.annotate('Polarized (x, z)', xy=(1.1, 0.75))
        plt.annotate('Kitaev (Toric)', xy=(0.2, 0.05))
        # plt.legend()
        plt.show()

    def phase_transition_border_predict_plzd_plot(self):

        plt.figure(figsize=(8, 8))
        data = self.phase_transition_border_predict_data('plzx')
        plt.plot(data['vertical_cut']['x'], data['vertical_cut']['y'])

        data = self.phase_transition_border_predict_data('plzz')
        plt.plot(data['horizontal_cut']['x'], data['horizontal_cut']['y'])

        plt.xlim([0, 1.49])
        plt.ylim([0, 1.49])
        plt.tick_params(direction='in')
        plt.xlabel('$h_z$')
        plt.ylabel('$h_x$')
        plt.annotate('Polarized (x)', xy=(0.4, 0.9))
        plt.annotate('Polarized (z)', xy=(1.1, 0.2))
        plt.annotate('Polarized (x, z)', xy=(1.1, 0.75))
        plt.annotate('Kitaev (Toric)', xy=(0.2, 0.05))
        # plt.legend()
        plt.show()

    def ae_phase_diagram_toric(self):

        plt.figure(figsize=(8, 8))
        matrix = self.predict_phase_profile('toric')
        im = plt.imshow(matrix['data_matrix_nrm'], cmap='twilight', origin='lower', interpolation='nearest', alpha=1)
        plt.xlabel('$h_z$')
        plt.ylabel('$h_x$')
        plt.tick_params(direction='in')
        plt.xticks([i for i in np.arange(0, 150, 15)], labels=[i / 100 for i in np.arange(0, 150, 15)])
        plt.yticks([i for i in np.arange(0, 150, 15)], labels=[i / 100 for i in np.arange(0, 150, 15)])
        plt.colorbar(im, label='loss', fraction=0.046, pad=0.04)
        plt.savefig('diagram_phase_toric_ae.pdf', format='pdf')
        plt.show()

    def mag_phase_diagram_toric(self):
        plt.figure(figsize=(8, 8))
        matrix = self.physics_profile('mag_value')
        im = plt.imshow(matrix['data_matrix'], cmap='twilight', origin='lower', interpolation='nearest',
                        alpha=1)
        plt.xlabel('$h_z$')
        plt.ylabel('$h_x$')
        plt.tick_params(direction='in')
        plt.xticks([i for i in np.arange(0, 150, 15)], labels=[i / 100 for i in np.arange(0, 150, 15)])
        plt.yticks([i for i in np.arange(0, 150, 15)], labels=[i / 100 for i in np.arange(0, 150, 15)])
        plt.colorbar(im, label='$\langle |m| \\rangle$', fraction=0.046, pad=0.04)
        plt.savefig('diagram_phase_toric_mag.pdf', format='pdf')
        plt.show()

    def ae_stream_plot(self, phase_name: str, data_type: str):

        matrix = self.predict_phase_profile(phase_name, data_type)

        # 1D arrays
        feature_x = np.arange(0, 150, 1)
        feature_y = np.arange(0, 150, 1)

        # Creating 2-D grid of features
        x, y = np.meshgrid(feature_x, feature_y)

        # Depict illustration
        plt.figure(figsize=(8, 8))
        ax = plt.axes()
        ax.set_facecolor("#e4e4e4")

        # plt.quiver(x, y, matrix['grd_ii_x'], matrix['grd_ii_y'])
        plt.imshow(matrix['data_matrix_nrm'], cmap='viridis', origin='lower', interpolation='nearest')
        # plt.plot([i for i in range(150)], matrix['data_matrix_nrm'][0, :])
        # plt.streamplot(x, y, matrix['grd_ii_x'], matrix['grd_ii_y'],
        #                density=3.2,
        #                linewidth=matrix['data_matrix_nrm'] + 0.50,
        #                color=matrix['data_matrix_nrm'],
        #                cmap='winter',
        #                integration_direction='both',
        #                broken_streamlines=True)
        plt.colorbar(label='loss', fraction=0.046, pad=0.04)

        # data = self.phase_transition_border_energy_data()
        #
        # plt.plot(
        #     np.array(data['vertical_cut']['x'][:8])*100,
        #     np.array(data['vertical_cut']['y'][:8])*100,
        #     linestyle='--', color='red')
        # plt.plot(
        #     np.array(data['horizontal_cut']['x'][:10])*100,
        #     np.array(data['horizontal_cut']['y'][:10])*100,
        #     linestyle='--', color='red')

        plt.xlabel('$h_z$')
        plt.ylabel('$h_x$')
        plt.tick_params(direction='in')
        plt.xticks([i for i in np.arange(0, 150, 15)], labels=[i / 100 for i in np.arange(0, 150, 15)])
        plt.yticks([i for i in np.arange(0, 150, 15)], labels=[i / 100 for i in np.arange(0, 150, 15)])
        # plt.xticks([i for i in range(0, 151, 10)], [i / 100 for i in range(0, 151, 10)])
        # plt.yticks([i for i in range(0, 151, 10)], [i / 100 for i in range(0, 151, 10)])

        # plt.annotate('Polarized', xy=(75, 75), color='black',
        #              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))
        # plt.annotate('Kitaev (Toric)', xy=(10, 5), color='black',
        #              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=1))

        plt.savefig('ea_stream_plot_toric.pdf', format='pdf')
        plt.show()

    def chishod(self):

        plt.figure(figsize=(8, 8))

        energy = self.physics_profile('energy')['data_matrix']
        mag = self.physics_profile('mag_value')['data_matrix']
        plt.imshow(mag, origin='lower', interpolation='nearest', alpha=1)

        # matrix = self.predict_phase_profile('plzx', 'lambda')
        # matrix = self.predict_phase_profile('plzz', 'lambda')
        matrix = self.predict_phase_profile('toric', 'lambda')
        # plt.imshow(matrix['data_matrix_nrm'], cmap='twilight', origin='lower', interpolation='nearest', alpha=1)
        # plt.imshow(matrix['grd_i_x_nrm'], cmap='twilight', origin='lower', interpolation='nearest', alpha=1)
        # plt.imshow(matrix['grd_i_y_nrm'], cmap='plasma', origin='lower', interpolation='nearest', alpha=1)
        plt.xlabel('$h_z$')
        plt.ylabel('$h_x$')
        plt.xticks([i for i in np.arange(0, 150, 15)], labels=[i / 100 for i in np.arange(0, 150, 15)])
        plt.yticks([i for i in np.arange(0, 150, 15)], labels=[i / 100 for i in np.arange(0, 150, 15)])
        plt.colorbar(label='loss', fraction=0.046, pad=0.04)

        # feature_x = np.arange(0, 150, 1)
        # feature_y = np.arange(0, 150, 1)
        #
        # # Creating 2-D grid of features
        # [x, y] = np.meshgrid(feature_x, feature_y)
        # contours = plt.contour(x, y, matrix['data_matrix_nrm'], 20, colors='red')
        # plt.clabel(contours, inline=True, fontsize=8)

        # plt.plot(matrix['grd_i_x_nrm'][0, :])
        # plt.plot(matrix['grd_i_x_nrm'][1, :])
        # plt.plot(matrix['grd_i_x_nrm'][3, :])
        # plt.plot(matrix['grd_i_x_nrm'][4, :])

        plt.savefig('diagram_phase_toric.pdf', format='pdf')
        plt.show()

    def sigmoid_plot(self):
        x = [i / 10 for i in range(-40, 40)]
        y = np.exp(x) / (1 + np.exp(x))
        y = (2 * y) - 1
        plt.plot(x, y)
        plt.axvline(x=0,
                    linestyle='dashed',
                    color='gray')
        plt.axhline(y=0,
                    linestyle='dashed',
                    color='gray')
        plt.tick_params(direction='in')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.savefig('bipolar_sigmoid.pdf', format='pdf')
        # plt.show()
