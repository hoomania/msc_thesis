from torch.utils.data import DataLoader
from tqdm import tqdm
import mdl_autoencoder as ae
import dataset_maker as dsm
import torch
import torch.utils
import torch.distributions
import numpy as np
import pandas as pd


class Training:
    def __init__(self,
                 model_save_to: str,
                 feature_length: int):
        self.model_save_to = model_save_to
        self.feature_length = feature_length

    def train_model(self,
                    data_path: str,
                    epoch_num: int,
                    batch_size: int,
                    lr: float,
                    weight_decay: float):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dataset = dsm.DatasetMaker(
            path=data_path,
            row_length=self.feature_length,
            model_type='ae')

        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True)

        model = ae.AutoEncoder(self.feature_length)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            params=model.to(device).parameters(),
            lr=lr,
            weight_decay=weight_decay)

        best_loss = np.inf
        for epoch in range(epoch_num):
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for _, (inputs, y, u, v) in loop:
                # Select device (if GPU is available)
                inputs = inputs.to(device)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                # Forward pass: Compute predicted y by passing x to the model
                encodes, outputs = model(inputs)
                # Compute loss
                loss = criterion(outputs, inputs)
                # Perform a backward pass (backpropagation)
                loss.backward()
                # Update the parameters
                optimizer.step()

                loop.set_description(f"Epoch [{str.zfill(str(epoch + 1), 2)}/{epoch_num}]")
                loop.set_postfix(loss=loss.item())
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), self.model_save_to)

        print(f'Best loss: {best_loss}')

    def evaluate_model(self,
                       data_path: str,
                       predict_save_to: str) -> None:
        dataset = dsm.DatasetMaker(
            path=data_path,
            row_length=self.feature_length,
            model_type='ae')

        test_loader = DataLoader(
            dataset=dataset,
            shuffle=False,
            num_workers=2)

        model = ae.AutoEncoder(self.feature_length)

        model.load_state_dict(torch.load(self.model_save_to))
        criterion = torch.nn.MSELoss()

        arr_pure_predict = []
        with torch.no_grad():
            loop = tqdm(enumerate(test_loader), total=len(test_loader))
            for _, (inputs, labels, v, u) in loop:
                # calculate output by running through the network
                encodes, outputs = model(inputs)
                # get the predictions
                arr_pure_predict.extend([[
                    criterion(outputs, inputs).item(),
                    round(v.item(), 2),
                    round(u.item(), 2)]])
                loop.set_description(f"Evaluating Model")

        dataframe = pd.DataFrame(arr_pure_predict, columns=['loss', 'v', 'u'])
        dataframe.to_csv(
            path_or_buf=f'./predict/{predict_save_to}',
            index=False)
