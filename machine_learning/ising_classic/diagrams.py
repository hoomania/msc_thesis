import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
font = {'family': 'cmr10',
        'size': 16}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 300

class Diagram:
    def __init__(self,
                 data_path: str,
                 lattice_length: int):
        self.data_path = data_path
        self.lattice_length = lattice_length

    def critical_temp(self):
        data = pd.read_csv(self.data_path)

        left_temp = data.query(f'temp <= 2.269')['temp'].unique()[-1]
        right_temp = data.query(f'temp > 2.269')['temp'].unique()[0]
        # left_temp = data.query(f'temp <= 2.340')['temp'].unique()[-1]
        # right_temp = data.query(f'temp > 2.340')['temp'].unique()[0]
        fm_dots = [
            [left_temp, right_temp],
            [np.average(data.query(f'temp == {left_temp}')['phase_f']),
             np.average(data.query(f'temp == {right_temp}')['phase_f'])]
        ]
        pm_dots = [
            [left_temp, right_temp],
            [np.average(data.query(f'temp == {left_temp}')['phase_p']),
             np.average(data.query(f'temp == {right_temp}')['phase_p'])]
        ]

        fm_slope = (fm_dots[1][1] - fm_dots[1][0]) / (fm_dots[0][1] - fm_dots[0][0])
        pm_slope = (pm_dots[1][1] - pm_dots[1][0]) / (pm_dots[0][1] - pm_dots[0][0])
        fm_b = fm_dots[1][1] - (fm_slope * fm_dots[0][1])
        pm_b = pm_dots[1][1] - (pm_slope * pm_dots[0][1])

        return (pm_b - fm_b) / (fm_slope - pm_slope)

    def phase_probability_boxplot(self, phase_name: str):
        data = pd.read_csv(self.data_path)
        temp_list = data['temp'].unique()

        weight_list = []
        for temp in temp_list:
            weight_list.append(data.query(f'temp == {temp}')[phase_name])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(weight_list, 0, '')
        ax.set(
            # title='Probability of ' + 'FM' if phase_name == 'phase_f' else 'PM ' + f'phase (L={self.lattice_length})',
            title=f'L={self.lattice_length}',
            xlabel='Temperature (K)',
            ylabel='Probability',
            xticks=(np.arange(len(temp_list)) + 1),
            xticklabels=temp_list)

        cff = (len(temp_list) - 1) / (temp_list[np.argmax(temp_list)] - temp_list[np.argmin(temp_list)])
        plt.axvline(
            x=(1.269 * cff) + 1,
            color='gray',
            linestyle='dotted',
            # label=u'H\u2082SO\u2084')
            label='$T_c$')
        plt.tick_params(direction='in')
        plt.locator_params(axis='x', nbins=15)
        plt.legend()
        plt.show()
        # plt.savefig(f'boxplot_{phase_name}_cnn_{self.lattice_length}.pdf', format='pdf')

    def phases_probability(self):
        data = pd.read_csv(self.data_path)
        min_value = data[['phase_f', 'phase_p']].min(numeric_only=True).min()
        max_value = data[['phase_f', 'phase_p']].max(numeric_only=True).max()

        data['phase_f'] = (data['phase_f'] - min_value) / (max_value - min_value)
        data['phase_p'] = (data['phase_p'] - min_value) / (max_value - min_value)

        temp_list = data['temp'].unique()
        weight_list_fm = []
        weight_list_pm = []
        for temp in temp_list:
            weight_list_fm.append(np.average(data.query(f'temp == {temp}')['phase_f']))
            weight_list_pm.append(np.average(data.query(f'temp == {temp}')['phase_p']))

        plt.plot(temp_list, weight_list_fm,
                 marker='o',
                 label='$Phase \,\, FM$')
        plt.plot(temp_list, weight_list_pm,
                 marker='o',
                 label='$Phase \,\, PM$')
        plt.ylim([-0.1, 1.1])
        plt.axvline(
            x=2.269,
            linestyle='dashed',
            color='green',
            label='$T_c \, (Analytical)$')
        plt.axvline(
            x=self.critical_temp(),
            linestyle='dotted',
            color='gray',
            label='$T_c \, (Machine)$')
        plt.tick_params(direction='in')
        plt.title(
            # f'L={self.lattice_length}')
            f'L={self.lattice_length}, $\Delta T_c = {self.critical_temp() - 2.269:.2}$')
            # f'Weights of output layer (L={self.lattice_length}, $\Delta T_c = {self.critical_temp() - 2.269:.2}$)')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Phase Probability')
        plt.legend()
        # plt.show()
        plt.savefig(f'output_layer_cnn_{self.lattice_length}.pdf', format='pdf')
