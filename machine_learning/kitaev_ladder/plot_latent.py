# import dataset_maker as dsm
# import pandas as pd
# import pickle as pkl
# import matplotlib.pyplot as plt
# import torch
#
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
# def plot_2d(
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
#     for i, (x, y) in enumerate(data):
#         z = model.encoder(x.to(device))
#         z = z.to('cpu').detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10', s=1)
#
#     plt.colorbar()
#     plt.show()
#
#
# def plot_3d(
#         model_file_path: str,
#         data_file_path: str,
#         feature_len: int,
#         fig_dpi: int = 200):
#
#     data = prepare_data(
#         data_file_path,
#         feature_len
#     )
#
#     plt.rcParams['figure.dpi'] = fig_dpi
#     ax = plt.axes(projection='3d')
#
#     with open(model_file_path, 'rb') as handle:
#         model = pkl.load(handle)
#
#     x_data = []
#     y_data = []
#     z_data = []
#
#     for i, (x, y) in enumerate(data):
#         z = model.encoder(x.to(device))
#         z = z.to('cpu').detach().numpy()
#         ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=y, cmap='rainbow', s=1)
#         x_data.extend(z[:, 0])
#         y_data.extend(z[:, 1])
#         z_data.extend(z[:, 2])
#
#         ax.view_init(45, 90)
#         plt.show()
#
#
# def save_data_plot_3d(
#         model_file_path,
#         data_file_path: str,
#         feature_len: int):
#
#     data = prepare_data(
#         data_file_path,
#         feature_len
#     )
#
#     with open(model_file_path, 'rb') as handle:
#         model = pkl.load(handle)
#
#     x_data = []
#     y_data = []
#     z_data = []
#
#     for i, (x, y) in enumerate(data):
#         z = model.encoder(x.to(device))
#         z = z.to('cpu').detach().numpy()
#         x_data.extend(z[:, 0])
#         y_data.extend(z[:, 1])
#         z_data.extend(z[:, 2])
#
#     df = pd.DataFrame({'x': x_data, 'y': y_data, 'z': z_data})
#     df.to_csv(
#         f'3d_plot_data.csv',
#         header=False,
#         index=False)
