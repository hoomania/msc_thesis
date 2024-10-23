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


class CheckData:

    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.data.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
        self.length = int(np.sqrt(self.data.shape[1]))
        self.n_sites = self.length ** 2

    def mag_chi_temp_dataframe(self) -> pd.DataFrame:
        data = self.data.drop(['temp', 'phase'], axis=1)
        m = data.sum(axis=1) / self.n_sites
        m2 = ((data * data).sum(axis=1)) / self.n_sites
        temp = self.data['temp']
        chi = (m2 - m ** 2) / temp

        return pd.DataFrame({'m': m, 'chi': chi, 'temp': temp})

    def tc(self) -> float:
        data = self.mag_chi_temp_dataframe()
        temp_list = data['temp'].unique()

        chi_mean = []
        for temp in temp_list:
            chi_mean.append(np.average(data.query(f'temp == {temp}')['chi']))

        return temp_list[np.argmax(chi_mean)]

    def magnetic_susceptibility(self):
        data = self.mag_chi_temp_dataframe()
        temp_list = data['temp'].unique()
        chi_list = []
        for temp in temp_list:
            chi_list.append(data.query(f'temp == {temp}')['chi'])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(chi_list, 0, '')
        ax.set(
            # title='Magnetic Susceptibility',
            title=f'L={self.length}',
            xlabel='Temperature (K)',
            ylabel='$\chi$',
            xticks=(np.arange(len(temp_list)) + 1),
            xticklabels=temp_list)

        cff = (len(temp_list) - 1) / (temp_list[np.argmax(temp_list)] - temp_list[np.argmin(temp_list)])
        plt.axvline(
            x=(1.269 * cff) + 1,
            color='gray',
            linestyle='dotted',
            label='$T_c$')
        plt.tick_params(direction='in')
        plt.locator_params(axis='x', nbins=15)
        plt.legend()
        # plt.show()
        plt.savefig(f'magsus_{self.length}.pdf', format='pdf')

    def magnetization(self):
        data = self.mag_chi_temp_dataframe()
        temp_list = data['temp'].unique()
        m_list = []
        for temp in temp_list:
            m_list.append(np.abs(data.query(f'temp == {temp}')['m']))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(m_list, 0, '')
        ax.set(
            title=f'L={self.length}',
            # title='Magnetization ',
            xlabel='Temperature (K)',
            ylabel='$\langle |m| \\rangle$',
            xticks=(np.arange(len(temp_list)) + 1),
            xticklabels=temp_list)

        cff = (len(temp_list) - 1) / (temp_list[np.argmax(temp_list)] - temp_list[np.argmin(temp_list)])
        plt.axvline(
            x=(1.269 * cff) + 1,
            color='gray',
            linestyle='dotted',
            label='$T_c$')
        plt.tick_params(direction='in')
        # plt.title(label='Magnetization', fontsize=24, font=font_name)
        plt.locator_params(axis='x', nbins=15)
        plt.legend()
        # plt.show()
        plt.savefig(f'mag_{self.length}.pdf', format='pdf')

    def avg_mag_sus(self):
        data = self.mag_chi_temp_dataframe()
        temp_list = data['temp'].unique()

        chi_avg = []
        m_avg = []
        for temp in temp_list:
            chi_avg.append(np.average(data.query(f'temp == {temp}')['chi']))
            m_avg.append(np.average(np.abs(data.query(f'temp == {temp}')['m'])))

        fig, ax = plt.subplots(figsize=(10, 6))

        g1 = plt.scatter(temp_list, m_avg, color='#ff7f0e', marker='$\\bigoplus$')
        # plt.title('Avg. of Magnetization and Magnetization Susceptibility')
        plt.title(f'L={self.length}')
        plt.xlabel('Temperature (K)')
        plt.locator_params(axis='x', nbins=15)
        plt.tick_params(direction='in')
        plt.xticks(np.arange(temp_list[np.argmin(temp_list)], temp_list[np.argmax(temp_list)] + 1, 0.1))
        plt.ylabel('$\langle |m| \\rangle$')

        ax.twinx()
        g2 = plt.scatter(temp_list, chi_avg, color='#1f77b4', marker='o')
        plt.locator_params(axis='x', nbins=15)
        plt.tick_params(direction='in')
        plt.ylabel('$\chi$')

        fig.tight_layout()
        g3 = plt.axvline(x=2.269, color='gray', linestyle='dotted')
        plt.tick_params(direction='in')
        ax.legend([g1, g2, g3], ['$\langle |m| \\rangle$', '$\chi$', '$T_c$'], bbox_to_anchor=[0.2, 0.6])
        plt.locator_params(axis='x', nbins=15)

        # plt.show()
        plt.savefig(f'mag_magsus_{self.length}.pdf', format='pdf')

    def lattice_snapshot(self):
        data = self.mag_chi_temp_dataframe()
        temp_list = data['temp'].unique()
        min_t = temp_list[np.argmin(temp_list)]
        max_t = temp_list[np.argmax(temp_list)]
        tc = self.tc()

        lattice_t_gt_tc = self.data.query(f'temp == {max_t}').sample().drop(
            ['temp', 'phase'],
            axis=1
        ).values.reshape((self.length, self.length))

        lattice_t_eq_tc = self.data.query(f'temp == {tc}').sample().drop(
            ['temp', 'phase'],
            axis=1
        ).values.reshape((self.length, self.length))

        lattice_t_lt_tc = self.data.query(f'temp == {min_t}').sample().drop(
            ['temp', 'phase'],
            axis=1).values.reshape((self.length, self.length))

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
        c = mpl.colors.ListedColormap(['#cecece', 'white', 'black'])
        n = mpl.colors.Normalize(vmin=-1, vmax=1)

        axes[0].matshow(lattice_t_gt_tc, cmap=c, norm=n)
        axes[0].set_title(f"$T = {max_t}K > T_c$")
        axes[1].matshow(lattice_t_eq_tc, cmap=c, norm=n)
        axes[1].set_title(f"$L = {self.length}$" + "\n\n" + f"$T = {tc}K \\approx T_c$")
        axes[2].matshow(lattice_t_lt_tc, cmap=c, norm=n)
        axes[2].set_title(f"$T = {min_t}K < T_c$")
        for i in range(3):
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        # plt.show()
        plt.savefig(f'lattice_snapshot_{self.length}.pdf', format='pdf')

    def lattice_brief_view(self):
        plt.imshow(self.data.drop(['temp', 'phase'], axis=1), origin='lower', interpolation='nearest')
        plt.title(f"\nSystem Evolution")
        plt.ylabel("Samples")
        plt.xticks([])
        plt.show()
