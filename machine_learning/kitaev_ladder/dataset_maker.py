import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetMaker(Dataset):

    def __init__(self,
                 path: str,
                 row_length: int,
                 model_type: str):
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x_data = torch.from_numpy(xy[:, :row_length])
        if model_type == 'fc':
            self.y_data = torch.nn.functional.one_hot(torch.LongTensor(xy[:, -1])).type(torch.FloatTensor)
            self.hx_data = xy[:, -3]
            self.hz_data = xy[:, -2]
        else:
            self.y_data = torch.from_numpy(xy[:, -1])
            self.hx_data = xy[:, -2]
            self.hz_data = xy[:, -1]

    def __getitem__(self, index):
        return self.x_data[index], \
               self.y_data[index], \
               self.hx_data[index], \
               self.hz_data[index]

    def __len__(self):
        return self.n_samples
