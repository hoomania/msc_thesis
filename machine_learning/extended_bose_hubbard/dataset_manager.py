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


def lambda_ae_dataset():
    data = pd.read_csv(f'../../data_set/extended_bose_hubbard/lambda_D16_all_labels.csv',
                       header=0,
                       index_col=False)

    mi_phase = data.query('u > 5.5 and u < 6 and v > 0 and v < 0.25').drop(
        columns=['energy', 'bond_entropy', 'round_bond_entropy'])
    mi_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/lambda_D16_mi_phase.csv', index=False)

    sf_phase = data.query('u > 1 and u < 1.5 and v > 0 and v < 0.25').drop(
        columns=['energy', 'bond_entropy', 'round_bond_entropy'])
    sf_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/lambda_D16_sf_phase.csv', index=False)

    hi_phase = data.query('u > 0.5 and u < 1 and v > 1.25 and v < 1.5').drop(
        columns=['energy', 'bond_entropy', 'round_bond_entropy'])
    hi_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/lambda_D16_hi_phase.csv', index=False)

    dw_phase = data.query('u > 0 and u < 0.5 and v > 3.5 and v < 3.75').drop(
        columns=['energy', 'bond_entropy', 'round_bond_entropy'])
    dw_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/lambda_D16_dw_phase.csv', index=False)


def physical_ae_dataset():
    data = pd.read_csv(f'../../data_set/extended_bose_hubbard/ae/physical_D16_all.csv',
                       header=0,
                       index_col=False).drop(
        columns=['b-energy-i', 'b-energy-j', 'ni', 'nj', 'n-total', 'n-average', 'E0', 'E-total', 'ad-i', 'ad-j',
                 'ni-nj', 'nj-ni', 'ad-i,aj',
                 'ad-j,ai', 'ai-aj', 'aj-ai'])

    mi_phase = data.query('u > 5.5 and u < 6 and v > 0 and v < 0.25')
    mi_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_mi_phase.csv', index=False)

    sf_phase = data.query('u > 1 and u < 1.5 and v > 0 and v < 0.25')
    sf_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_sf_phase.csv', index=False)

    hi_phase = data.query('u > 0.5 and u < 1 and v > 1.25 and v < 1.5')
    hi_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_hi_phase.csv', index=False)

    dw_phase = data.query('u > 0 and u < 0.5 and v > 3.5 and v < 3.75')
    dw_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_dw_phase.csv', index=False)


def physical_ae_normalized_dataset():
    data = pd.read_csv('../../data_set/extended_bose_hubbard/ae/physical_D16_all.csv',
                       header=0,
                       index_col=False)

    for colName in data.columns[:-2]:
        minim = data[colName].min(numeric_only=True)
        maxim = data[colName].max(numeric_only=True)
        data[colName] = (data[colName] - minim) / (maxim - minim)

    mi_phase = data.query('u > 5.5 and u < 6 and v > 0 and v < 0.25')
    mi_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_mi_phase.csv', index=False)

    sf_phase = data.query('u > 1 and u < 1.5 and v > 0 and v < 0.25')
    sf_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_sf_phase.csv', index=False)

    hi_phase = data.query('u > 0.5 and u < 1 and v > 1.25 and v < 1.5')
    hi_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_hi_phase.csv', index=False)

    dw_phase = data.query('u > 0 and u < 0.5 and v > 3.5 and v < 3.75')
    dw_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_dw_phase.csv', index=False)


def physical_ae_pca_dataset():
    data = pd.read_csv(f'../../data_set/extended_bose_hubbard/ae/physical_D16_all.csv',
                       header=0,
                       index_col=False)

    u_v = data.iloc[:, -2:]
    feature = np.array(data.drop(columns=['v', 'u']))

    pca_data = pd.DataFrame(PCA().fit_transform(feature))

    data = pca_data.assign(v=np.array(u_v.iloc[:, 0]), u=np.array(u_v.iloc[:, 1]))

    mi_phase = data.query('u > 5.5 and u < 6 and v > 0 and v < 0.25')
    mi_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_mi_phase.csv', index=False)

    sf_phase = data.query('u > 1 and u < 1.5 and v > 0 and v < 0.25')
    sf_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_sf_phase.csv', index=False)

    hi_phase = data.query('u > 0.5 and u < 1 and v > 1.25 and v < 1.5')
    hi_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_hi_phase.csv', index=False)

    dw_phase = data.query('u > 0 and u < 0.5 and v > 3.5 and v < 3.75')
    dw_phase.to_csv(path_or_buf='../../data_set/extended_bose_hubbard/ae/physical_D16_dw_phase.csv', index=False)


# lambda_ae_dataset()
# physical_ae_dataset()
# physical_ae_pca_dataset()
physical_ae_normalized_dataset()
