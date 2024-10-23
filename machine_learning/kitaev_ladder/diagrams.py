import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
font = {'family': 'cmr10',
        'size': 12}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 100


class Diagram:
    def __init__(self,
                 virtual_dimension: int):
        self.virtual_dimension = virtual_dimension

    def phase_transition_point(self,
                               data: pd.DataFrame):
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

    def phases_probability(self,
                           data: pd.DataFrame,
                           x_title: str):
        x_values = data['x'].unique()

        plt.plot(x_values, data['phase_toric'],
                 marker='o',
                 label='Toric Phase')
        plt.plot(x_values, data['phase_fm'],
                 marker='o',
                 label='Polarized Phase')
        plt.ylim([-0.1, 1.1])
        plt.axvline(
            x=0.55,
            linestyle='dashed',
            color='green',
            label='$h_c$')

        plt.axvline(
            x=self.phase_transition_point(data),
            linestyle='dotted',
            color='gray',
            label='$h_c \, (Machine)$')

        plt.tick_params(direction='in')
        plt.title(
            # f'Weights of output layer (D={self.virtual_dimension}, $h_c={self.phase_transition_point(data):0.2}$)')
            f'$\chi={self.virtual_dimension}, \Delta h_c={(self.phase_transition_point(data)-0.55):0.2}$')
        plt.xlabel(x_title)
        plt.ylabel('Phase Probability')
        plt.legend()
        # plt.show()
        plt.savefig(f'output_layer_toric_ladder_fc_physical_D32.pdf', format='pdf')

    def phase_diagram_2d(self,
                         data: pd.DataFrame):
        hz_list = data.sort_values('hz')['hz'].unique()

        minim = data['loss'].min(numeric_only=True)
        maxim = data['loss'].max(numeric_only=True)
        data['loss'] = (data['loss'] - minim) / (maxim - minim)
        data['loss'] = 1 - data['loss']
        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['loss'])

        data_matrix = np.array(data_matrix)
        plt.imshow(data_matrix, cmap='plasma_r', origin='lower', interpolation='nearest')
        plt.title(
            f'Phase Diagram (D={self.virtual_dimension})')
        plt.colorbar()

        feature_x = np.arange(0, 150, 1)
        feature_y = np.arange(0, 150, 1)

        # Creating 2-D grid of features
        [x, y] = np.meshgrid(feature_x, feature_y)
        contours = plt.contour(x, y, data_matrix, 10, colors='white')
        plt.clabel(contours, inline=True, fontsize=8)
        plt.show()

        y_list = [i / 100 for i in range(150)]
        hx_max_list = []
        hz_max_list = []
        for hx in range(150):
            hx_line = data.query(f'hz == {hx / 100}').sort_values('hx')['loss']
            # print(f'hz: {hx/100}, max probability: {np.max(hx_line)}, hz max: {np.argmax(hx_line)/100}')
            hx_max = np.argmax(hx_line) / 100
            if hx_max == 0:
                hx_max_list.append(np.argmin(np.gradient(hx_line)) / 100)
            else:
                hx_max_list.append(np.argmax(hx_line) / 100)

            hz_line = data.query(f'hx == {hx / 100}').sort_values('hz')['loss']
            hz_max = np.argmax(hz_line) / 100
            if hz_max == 0 or hx / 100 < 1.5:
                hz_max_list.append(np.argmin(np.gradient(hz_line)) / 100)
                # print(
                #     f'hx: {hx / 100}, max probability: {np.max(hz_line)}, hz max: {np.argmin(np.gradient(hz_line)) / 100}')
            else:
                hz_max_list.append(np.argmax(hz_line) / 100)
                # print(f'hx: {hx / 100}, max probability: {np.max(hz_line)}, hz max: {np.argmax(hz_line) / 100}')
            # plt.plot(y_list, np.array(hx_line), label=f'{hx / 100}')

        # plt.axhline(y=0.5)
        # plt.legend()
        # plt.show()

        # plt.plot(hx_max_list, y_list, 'o')
        # plt.plot(y_list, hz_max_list, 'o')
        # plt.grid()
        # plt.ylim([0, 1.5])
        # plt.show()
        #
        # plt.plot(y_list,
        #          np.array(data.query(f'hx == 0.53').sort_values('hz')['loss']), 'o')
        # plt.show()
        # plt.plot(y_list,
        #          np.gradient(np.array(data.query(f'hx == 0.53').sort_values('hz')['loss'])), 'o')
        # plt.plot(y_list,
        #          np.gradient(np.gradient(np.array(data.query(f'hx == 0.53').sort_values('hz')['loss']))), 'o')
        # plt.axhline(y=0)
        # plt.show()

        # hz0 = data.query(f'hz == 0.0').sort_values('hx')['loss']
        # plt.plot([i / 100 for i in range(150)], np.array(hz0), '-o')
        # plt.show()
        # plt.plot([i / 100 for i in range(150)], np.gradient(np.array(hz0)), '-o')
        # plt.show()

    def gradient_test(self,
                      data: pd.DataFrame):
        hz_list = data.sort_values('hz')['hz'].unique()

        minim = data['loss'].min(numeric_only=True)
        maxim = data['loss'].max(numeric_only=True)
        data['loss'] = (data['loss'] - minim) / (maxim - minim)
        data['loss'] = 1 - data['loss']
        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['loss'])

        data_matrix = np.array(data_matrix)
        grd_I_mtx = np.gradient(data_matrix)
        grd_I_x = grd_I_mtx[1]
        grd_I_y = grd_I_mtx[0]
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        # plt.imshow(grd_I_x, cmap='plasma', origin='lower', interpolation='nearest')
        # plt.title(
        #     f'Gradient(I) x (D={self.virtual_dimension})')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(grd_I_y, cmap='plasma', origin='lower', interpolation='nearest')
        # plt.title(
        #     f'Gradient(I) y (D={self.virtual_dimension})')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(grd_II_x.reshape(150, 150), cmap='twilight', origin='lower', interpolation='nearest')
        # plt.title(
        #     f'Gradient(II) x (D={self.virtual_dimension})')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(grd_II_y.reshape(150, 150), cmap='twilight', origin='lower', interpolation='nearest')
        # plt.title(
        #     f'Gradient(II) y (D={self.virtual_dimension})')
        # plt.colorbar()
        # plt.show()

        feature_x = np.arange(0, 150, 1)
        feature_y = np.arange(0, 150, 1)

        # Creating 2-D grid of features
        [x, y] = np.meshgrid(feature_x, feature_y)
        contours = plt.contour(x, y, np.array(data_matrix), [0.50, 0.60, 0.70, 0.80, 0.90], cmap='Wistia')
        plt.clabel(contours, inline=True, fontsize=8)

        contours = plt.contour(x, y, grd_II_x, 4, colors='white')
        print(contours.get_array())
        plt.clabel(contours, inline=True, fontsize=8)
        plt.imshow(np.array(data_matrix),
                   extent=[0, 150, 0, 150],
                   origin='lower',
                   cmap='plasma',
                   alpha=0.70)
        # plt.tick_params(left=False, bottom=False)

        # plt.plot([53.48, 69.42], [17.92, 32.01], linestyle='dashed', color='green')
        # plt.plot([33.21, 53.48, 69.42], [0, 17.92, 32.01], linestyle='dashed', color='green')
        # plt.plot([69.42, 150], [32.01, 103.24], linestyle='dotted', color='yellow')

        # plt.imshow(grd_II_y.reshape(150, 150),
        #            extent=[0, 150, 0, 150],
        #            origin='lower',
        #            cmap='plasma',
        #            # interpolation='nearest',
        #            alpha=0.25)

        plt.rcParams.update({
            "text.usetex": True,
            'font.sans-serif': 'Noto Serif Display'})
        plt.title(
            f'Kitaev Ladder Phase Diagram (D={self.virtual_dimension})',
            fontsize=16)
        plt.colorbar(label='Percent (1 - loss)')
        plt.xlabel('$h_x$', fontsize=16)
        plt.ylabel('$h_z$', fontsize=16)
        plt.xticks([i for i in range(0, 151, 10)], [i / 100 for i in range(0, 151, 10)])
        plt.yticks([i for i in range(0, 151, 10)], [i / 100 for i in range(0, 151, 10)])
        plt.show()

    def mag_diagram_2d(self,
                       data: pd.DataFrame):
        hz_list = data['hz'].unique()
        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['mag'])

        plt.imshow(np.array(data_matrix), cmap='twilight', origin='lower', interpolation='nearest')
        plt.title(
            f'Magnetization Diagram (D={self.virtual_dimension})')
        plt.colorbar()
        plt.show()

    def energy_diagram(self,
                       data: pd.DataFrame):
        hz_list = data.sort_values('hz')['hz'].unique()

        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['energy'])

        data_matrix = np.array(data_matrix)
        grd_I_mtx = np.gradient(data_matrix)
        grd_I_x = grd_I_mtx[1]
        grd_I_y = grd_I_mtx[0]
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        feature_x = np.arange(0, 150, 1)
        feature_y = np.arange(0, 150, 1)

        # Creating 2-D grid of features
        [x, y] = np.meshgrid(feature_x, feature_y)

        plt.figure(figsize=(12, 12))

        # plt.imshow(data_matrix, cmap='plasma', origin='lower', interpolation='nearest')
        # plt.title(
        #     f'Energy plot (D={self.virtual_dimension})')
        # plt.colorbar()
        # plt.show()

        # contours = plt.contour(x, y, grd_I_x, 10, colors='white')
        # plt.clabel(contours, inline=True, fontsize=8)


        contours = plt.contour(x, y, grd_I_x, [-0.036], colors='white')
        plt.clabel(contours, inline=True, fontsize=0)
        contours = plt.contour(x, y, grd_I_y, [-0.030], colors='white')
        plt.clabel(contours, inline=True, fontsize=0)
        plt.imshow(grd_I_x, cmap='plasma', origin='lower', interpolation='nearest')
        plt.title(
            f'Gradient(I) x (D={self.virtual_dimension})')
        plt.xlabel('$h_x$', fontsize=16)
        plt.ylabel('$h_z$', fontsize=16)
        plt.colorbar(label='Percent (1-loss)')
        plt.xticks([i for i in range(0, 151, 10)], [i / 100 for i in range(0, 151, 10)])
        plt.yticks([i for i in range(0, 151, 10)], [i / 100 for i in range(0, 151, 10)])
        # plt.colorbar()
        # plt.show()
        plt.imshow(grd_I_y, cmap='plasma', origin='lower', interpolation='nearest', alpha=0.5)
        plt.title(
            f'Gradient(I) y (D={self.virtual_dimension})')
        # plt.colorbar()
        plt.show()
        #
        # plt.imshow(grd_II_x.reshape(150, 150), cmap='twilight', origin='lower', interpolation='nearest')
        # plt.title(
        #     f'Gradient(II) x (D={self.virtual_dimension})')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(grd_II_y.reshape(150, 150), cmap='twilight', origin='lower', interpolation='nearest')
        # plt.title(
        #     f'Gradient(II) y (D={self.virtual_dimension})')
        # plt.colorbar()
        # plt.show()
        #
        # xx = grd_II_x*grd_II_x
        # yy = grd_II_y*grd_II_y
        # rslt = np.sqrt(xx + yy)
        #
        # plt.imshow(rslt.reshape(150, 150), cmap='twilight', origin='lower', interpolation='nearest')
        # plt.title(
        #     f'GRD II altitude (D={self.virtual_dimension})')
        # plt.colorbar()
        # plt.show()

    def countor_plot(self,
                     data: pd.DataFrame):
        feature_x = np.arange(0, 150, 1)
        feature_y = np.arange(0, 150, 1)

        # Creating 2-D grid of features
        [x, y] = np.meshgrid(feature_x, feature_y)

        hz_list = data.sort_values('hz')['hz'].unique()
        minim = data['loss'].min(numeric_only=True)
        maxim = data['loss'].max(numeric_only=True)
        data['loss'] = (data['loss'] - minim) / (maxim - minim)
        data['loss'] = 1 - data['loss']
        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['loss'])

        # plots contour lines
        contours = plt.contour(x, y, np.array(data_matrix), 3, colors='white')
        plt.clabel(contours, inline=True, fontsize=8)
        plt.imshow(np.array(data_matrix), extent=[0, 150, 0, 150], origin='lower',
                   cmap='plasma', alpha=0.75)
        plt.title('Normalized Loss Contour Plot')
        plt.xlabel('$h_x$')
        plt.ylabel('$h_z$')
        plt.colorbar(label='Percent (1-loss)')
        plt.show()

    def toric_phase_diagram(self,
                            data: pd.DataFrame):
        # data = pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_toric.csv')
        hz_list = data.sort_values('hz')['hz'].unique()

        minim = data['loss'].min(numeric_only=True)
        maxim = data['loss'].max(numeric_only=True)
        data['loss'] = (data['loss'] - minim) / (maxim - minim)
        data['loss'] = 1 - data['loss']
        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['loss'])

        data_matrix = np.array(data_matrix)
        grd_I_mtx = np.gradient(data_matrix)
        grd_I_x = grd_I_mtx[1]
        grd_I_y = grd_I_mtx[0]
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        return data_matrix, grd_I_x, grd_I_y, grd_II_x, grd_II_y

    def plzd_x_phase_diagram(self,
                            data: pd.DataFrame):
        # data = pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_plzd_x.csv')
        hz_list = data.sort_values('hz')['hz'].unique()

        minim = data['loss'].min(numeric_only=True)
        maxim = data['loss'].max(numeric_only=True)
        data['loss'] = (data['loss'] - minim) / (maxim - minim)
        data['loss'] = 1 - data['loss']
        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['loss'])

        data_matrix = np.array(data_matrix)
        grd_I_mtx = np.gradient(data_matrix)
        grd_I_x = grd_I_mtx[1]
        grd_I_y = grd_I_mtx[0]
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        return data_matrix, grd_I_x, grd_I_y, grd_II_x, grd_II_y

    def plzd_z_phase_diagram(self,
                            data: pd.DataFrame):
        # data = pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_plzd_z.csv')
        hz_list = data.sort_values('hz')['hz'].unique()

        minim = data['loss'].min(numeric_only=True)
        maxim = data['loss'].max(numeric_only=True)
        data['loss'] = (data['loss'] - minim) / (maxim - minim)
        data['loss'] = 1 - data['loss']
        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['loss'])

        data_matrix = np.array(data_matrix)
        grd_I_mtx = np.gradient(data_matrix)
        grd_I_x = grd_I_mtx[1]
        grd_I_y = grd_I_mtx[0]
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        return data_matrix, grd_I_x, grd_I_y, grd_II_x, grd_II_y

    def physics_profile(self, target_col: str):
        data = pd.read_csv(f'../../data_set/kitaev_ladder/physical_profile_D32_all_main.csv')
        hz_list = data.sort_values('hz')['hz'].unique()

        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')[target_col])

        data_matrix = np.array(data_matrix)
        grd_I_mtx = np.gradient(data_matrix)
        grd_I_x = grd_I_mtx[1]
        grd_I_y = grd_I_mtx[0]
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        return data_matrix, grd_I_x, grd_I_y, grd_II_x, grd_II_y

    def vector_plot(self,
                    data_toric: pd.DataFrame,
                    data_x_plzd: pd.DataFrame,
                    data_z_plzd: pd.DataFrame,
                    ):
        # hz_list = data.sort_values('hz')['hz'].unique()
        #
        # minim = data['loss'].min(numeric_only=True)
        # maxim = data['loss'].max(numeric_only=True)
        # data['loss'] = (data['loss'] - minim) / (maxim - minim)
        # data['loss'] = 1 - data['loss']
        # data_matrix = []
        # for hz in hz_list:
        #     data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['loss'])
        #
        # data_matrix = np.array(data_matrix)
        # grd_I_mtx = np.gradient(data_matrix)
        # grd_I_x = grd_I_mtx[1]
        # grd_I_y = grd_I_mtx[0]
        # grd_II_x = np.gradient(grd_I_x, axis=1)
        # grd_II_y = np.gradient(grd_I_y, axis=0)

        phase_toric = self.toric_phase_diagram(data_toric)
        phase_plzd_x = self.plzd_x_phase_diagram(data_x_plzd)
        phase_plzd_z = self.plzd_z_phase_diagram(data_z_plzd)
        energy_profile = self.physics_profile('energy')


        # 1D arrays
        feature_x = np.arange(0, 150, 1)
        feature_y = np.arange(0, 150, 1)

        # Creating 2-D grid of features
        x, y = np.meshgrid(feature_x, feature_y)

        # Depict illustration
        plt.figure(figsize=(12, 12))
        ax = plt.axes()
        ax.set_facecolor("#e4e4e4")

        # plt.quiver(x, y, phase_toric[3], phase_toric[4])
        plt.streamplot(x, y, phase_toric[1], phase_toric[2],
                       density=3.2,
                       linewidth=phase_toric[0]+0.50,
                       color=phase_toric[0],
                       cmap='winter')
        plt.colorbar(label='Percent (1-loss)')

        plt.axvline(x=55)
        plt.axhline(y=16)
        plt.contour(x, y, energy_profile[3], [-0.0305], colors='red')
        # # plt.clabel(contours, inline=True, fontsize=8)
        plt.contour(x, y, energy_profile[4], [-0.0201], colors='red')
        # # plt.clabel(contours, inline=True, fontsize=8)
        #
        # plt.imshow(phase_plzd_x[0],
        #            extent=[0, 150, 0, 150],
        #            origin='lower',
        #            cmap='copper',
        #            alpha=0.0)
        #
        # plt.imshow(phase_plzd_z[0],
        #            extent=[0, 150, 0, 150],
        #            origin='lower',
        #            cmap='copper',
        #            alpha=0.0)


        # plt.colorbar()

        plt.title('Second Gradient of Prediction Data (Stream Plot)')

        plt.xlabel('$h_x$', fontsize=16)
        plt.ylabel('$h_z$', fontsize=16)
        plt.xticks([i for i in range(0, 151, 10)], [i / 100 for i in range(0, 151, 10)])
        plt.yticks([i for i in range(0, 151, 10)], [i / 100 for i in range(0, 151, 10)])
        # Show plot with grid
        # plt.grid()
        plt.show()

    def three_dim_plot(self):

        data = pd.read_csv(f'../../data_set/kitaev_ladder/physical_profile_D32_all_main.csv')
        hz_list = data.sort_values('hz')['hz'].unique()

        minim = data['mag_value'].min(numeric_only=True)
        maxim = data['mag_value'].max(numeric_only=True)
        data['mag_value'] = (data['mag_value'] - minim) / (maxim - minim)
        # data['mag_value'] = 1 - data['mag_value']
        mgz_surface = []
        for hz in hz_list:
            mgz_surface.append(data.query(f'hz == {hz}').sort_values('hx')['mag_value'])

        mgz_surface = np.array(mgz_surface)
        # energy_profile = self.physics_profile('mag_value')
        # energy_profile = self.toric_phase_diagram()
        feature_x = np.arange(0, 150, 1)
        feature_y = np.arange(0, 150, 1)

        # Creating 2-D grid of features
        x, y = np.meshgrid(feature_x, feature_y)


        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        ax.plot_surface(x, y, mgz_surface, cmap='cool', alpha=0.8)
        ax.plot_wireframe(x, y, mgz_surface, cmap='cool', alpha=0.8)

        # ax.plot_surface(x, y, grd_I_y, cmap='cool', alpha=0.8)
        # ax.plot_wireframe(x, y, grd_I_y, cmap='cool', alpha=0.8)
        # ax.plot_wireframe(x, y, grd_II_x, cmap='cool', alpha=0.8)
        # ax.plot_wireframe(x, y, grd_II_y, cmap='cool', alpha=0.8)

        ax.set_title('Magnetization', fontsize=14)
        ax.set_xlabel('$h_x$', fontsize=12)
        ax.set_ylabel('$h_z$', fontsize=12)
        ax.set_zlabel('mag', fontsize=12)

        plt.show()

    def surface_inner_product(self):
        data = pd.read_csv(f'../../data_set/kitaev_ladder/physical_profile_D32_all_main.csv')
        hz_list = data.sort_values('hz')['hz'].unique()

        minim = data['mag_value'].min(numeric_only=True)
        maxim = data['mag_value'].max(numeric_only=True)
        data['mag_value'] = (data['mag_value'] - minim) / (maxim - minim)
        # data['mag_value'] = 1 - data['mag_value']
        mgz_surface = []
        for hz in hz_list:
            mgz_surface.append(data.query(f'hz == {hz}').sort_values('hx')['mag_value'])

        mgz_surface = np.array(mgz_surface).reshape(150*150)
        machine_surface = self.toric_phase_diagram()[0].reshape(150*150)

        inner = np.inner(mgz_surface, machine_surface)
        norm_mgz = np.linalg.norm(mgz_surface)
        norm_machine = np.linalg.norm(machine_surface)

        cosine_theta = inner / (norm_mgz * norm_machine)
        degree = (90 * np.arccos(cosine_theta)) / np.pi
        print(cosine_theta, degree)

    def three_dim_stream_plot(self,
                              data: pd.DataFrame):
        hz_list = data.sort_values('hz')['hz'].unique()

        data_matrix = []
        for hz in hz_list:
            data_matrix.append(data.query(f'hz == {hz}').sort_values('hx')['loss'])

        data_matrix = np.array(data_matrix)
        grd_I_mtx = np.gradient(data_matrix)
        grd_I_x = grd_I_mtx[1]
        grd_I_y = grd_I_mtx[0]
        grd_II_x = np.gradient(grd_I_x, axis=1)
        grd_II_y = np.gradient(grd_I_y, axis=0)

        feature_x = np.arange(0, 150, 1)
        feature_y = np.arange(0, 150, 1)
        # feature_z = np.arange(0, 1, 0.1)

        # Creating 2-D grid of features
        x, y = np.meshgrid(feature_x, feature_y)
        z = np.array(data_matrix)
        ax = plt.figure().add_subplot(projection='3d')

        # Make the grid
        # x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
        #                       np.arange(-0.8, 1, 0.2),
        #                       np.arange(-0.8, 1, 0.8))

        # Make the direction data for the arrows
        u = grd_II_x
        v = grd_II_y
        w = z

        ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
        # ax.streamplot3(x, y, z, u, v, w,  linewidth=2, headwidth=2, density=4, interval=20, pipe=True, radius=0.02)
        plt.show()
