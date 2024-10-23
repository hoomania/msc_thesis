from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataset_maker as dsm
import mdl_fc as fcm
import torch
import pandas as pd
import numpy as np


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
                    weight_decay: float) -> None:

        dataset = dsm.DatasetMaker(data_path, self.feature_length)
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2)

        model = fcm.FC(input_dim=self.feature_length)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay)

        best_loss = np.inf
        for epoch in range(epoch_num):
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for _, (inputs, labels, temp) in loop:
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(inputs)
                # Compute loss
                loss = criterion(y_pred, labels)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
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
        dataset = dsm.DatasetMaker(data_path, self.feature_length)
        test_loader = DataLoader(
            dataset=dataset,
            shuffle=False,
            num_workers=2)

        model = fcm.FC(input_dim=self.feature_length)
        model.load_state_dict(torch.load(self.model_save_to))

        arr_labels = []
        arr_predicts = []
        arr_pure_predict = []
        with torch.no_grad():
            for _, (inputs, labels, temp) in enumerate(test_loader):
                # calculate output by running through the network
                outputs = model(inputs)
                # get the predictions
                arr_pure_predict.extend([[
                    outputs.data[0][0].item(),
                    outputs.data[0][1].item(),
                    round(temp.item(), 2)]])
                # set report data
                arr_predicts.append(int(np.argmax(outputs.data[0])))
                arr_labels.append(int(np.argmax(labels[0])))

        with open(f'./predict/report_fc_L{self.feature_length}.txt', 'w') as report_file:
            print('Classification Report:\n' + classification_report(arr_labels, arr_predicts,
                                                                     target_names=['Ferromagnetism', 'Paramagnetism'],
                                                                     zero_division=True), file=report_file)
            print(f'Accuracy: {accuracy_score(arr_labels, arr_predicts):.2}', file=report_file)
            print(f'\nConfusion Matrix: \n{confusion_matrix(arr_labels, arr_predicts)}', file=report_file)

        dataframe = pd.DataFrame(arr_pure_predict, columns=['phase_f', 'phase_p', 'temp'])
        dataframe.to_csv(
            path_or_buf=f'./predict/{predict_save_to}',
            index=False)
