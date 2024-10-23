from mcmc import monte_carlo as mc
from colorama import Fore
from os.path import exists, abspath, dirname
import datetime
import numpy as np
import pandas as pd


class Sampling:

    def __init__(self,
                 lattice_length: int,
                 lattice_dim: int,
                 lattice_profile: str,
                 configs: int,
                 temp_low: float,
                 temp_high: float,
                 temp_step: float,
                 samples_per_temp: int = 10,
                 beta_inverse: bool = False,
                 path_output: str = None):

        self.lattice_length = lattice_length
        self.lattice_dim = lattice_dim
        self.lattice_profile = lattice_profile
        self.configs = configs
        self.temp_low = temp_low
        self.temp_high = temp_high
        self.temp_step = temp_step
        self.samples_per_temp = samples_per_temp
        self.beta_inverse = beta_inverse
        if path_output is None:
            self.path_output = dirname(abspath(__file__)) + '/data'
        else:
            self.path_output = path_output

    def take_sample(self):
        ts_sampling = datetime.datetime.now()
        print(f'\n---- Sampling Start ----')

        temperatures = np.linspace(
            self.temp_high, self.temp_low,
            int((self.temp_high - self.temp_low) / self.temp_step) + 1
        )

        row = []
        temp_value = []
        for i in range(self.configs):
            print(f'\n---- Configure {i} Started ----')
            ts_configs = datetime.datetime.now()
            mc_object = mc.MonteCarlo(self.lattice_length, self.lattice_dim, self.lattice_profile)
            for temp in temperatures:
                samples = mc_object.sampling(self.samples_per_temp, temp, self.beta_inverse)

                for sample in samples:
                    array = [j for j in sample]
                    row.append(array)
                    temp_value.append(temp)

            te_configs = datetime.datetime.now()
            print(Fore.MAGENTA + f'\n#### Configure {i} ####')
            print(f'   Time Start:  {ts_configs}')
            print(f'     Time End:  {te_configs}')
            print(f'Time Duration:  {te_configs - ts_configs}')
            print(f'---------------------')
            print(Fore.RESET)

        print(f'\n#### Saving Data Is In Progress ####')
        dataframe = pd.DataFrame(row)
        dataframe['Temp'] = temp_value
        if exists(f"{self.path_output}/data_{self.lattice_profile}_L{self.lattice_length}.csv"):
            stamp = int(datetime.datetime.now().timestamp())
            file_name = f'data_{self.lattice_profile}_L{self.lattice_length}_{stamp}.csv'
        else:
            file_name = f'data_{self.lattice_profile}_L{self.lattice_length}.csv'

        dataframe.to_csv(fr"{self.path_output}/{file_name}")
        print(f'\n#### Saving Data Finished ####')

        te_sampling = datetime.datetime.now()
        print(Fore.GREEN + f'\n#### Sampling Profile ####')
        print(f'             Time Start: {ts_sampling}')
        print(f'               Time End: {te_sampling}')
        print(f'          Time Duration: {te_sampling - ts_sampling}')
        print(f'\n        Lattice Profile: {self.lattice_profile}')
        print(f'         Lattice Length: {self.lattice_length}')
        print(f'      Lattice Dimension: {self.lattice_dim}')
        print(f'                Configs: {self.configs}')
        print(f'      Temperature Range: [{self.temp_low}, {self.temp_high}] (h={self.temp_step})')
        print(f'Samples Per Temperature: {self.samples_per_temp}')
        print(f'           Beta Inverse: {self.beta_inverse}')
        print(f'              File Name: {file_name}')
        print(f'--------------------------')

        print(Fore.RESET)

        print(f'\n---- Sampling End ----')
