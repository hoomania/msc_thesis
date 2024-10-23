from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataset_maker as dsm
import mdl_fc as mdl
import torch
import pandas as pd
import numpy as np


class Training:

    def __init__(self,
                 model_save_to: str,
                 virtual_dim: int,
                 feature_length: int,
                 hidden_layer_nodes: int,
                 classes_num: int = 3):
        self.model_save_to = model_save_to
        self.virtual_dim = virtual_dim
        self.hidden_layer_nodes = hidden_layer_nodes
        self.classes_num = classes_num
        self.feature_length = feature_length

    def train_model(self,
                    data_path: str,
                    epoch_num: int,
                    batch_size: int,
                    lr: float,
                    weight_decay: float) -> None:

        dataset = dsm.DatasetMaker(
            path=data_path,
            row_length=self.feature_length,
            model_type='fc')

        model = mdl.FC(
            input_dim=self.feature_length,
            output_dim=self.classes_num,
            hidden_layer=self.hidden_layer_nodes)

        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2)

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay)

        best_loss = np.inf
        for epoch in range(epoch_num):
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (inputs, labels, u, v) in loop:
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(inputs)
                # Compute loss
                loss = criterion(y_pred, labels)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                # perform a backward pass (backpropagation)
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
            model_type='fc')
        test_loader = DataLoader(
            dataset=dataset,
            shuffle=False,
            num_workers=2)

        model = mdl.FC(
            input_dim=self.feature_length,
            output_dim=self.classes_num,
            hidden_layer=self.hidden_layer_nodes)
        model.load_state_dict(torch.load(self.model_save_to))

        arr_labels = []
        arr_predicts = []
        arr_pure_predict = []
        with torch.no_grad():
            for _, (inputs, labels, u, v) in enumerate(test_loader):
                # calculate output by running through the network
                outputs = model(inputs)
                # get the predictions
                arr_pure_predict.extend([[
                    outputs.data[0][0].item(),
                    outputs.data[0][1].item(),
                    outputs.data[0][2].item()]])
                # set report data
                arr_predicts.append(int(np.argmax(outputs.data[0])))
                arr_labels.append(int(np.argmax(labels[0])))

        with open(f'./predict/report_fc_D{self.virtual_dim}.txt', 'w') as report_file:
            print('Classification Report:\n' + classification_report(arr_labels, arr_predicts,
                                                                     target_names=['Super Fluid', 'Matt Insulator',
                                                                                   'Density Wave'],
                                                                     zero_division=True), file=report_file)
            print(f'Accuracy: {accuracy_score(arr_labels, arr_predicts):.2}', file=report_file)
            print(f'\nConfusion Matrix: \n{confusion_matrix(arr_labels, arr_predicts)}', file=report_file)

        dataframe = pd.DataFrame(arr_pure_predict, columns=['phase_sf', 'phase_mi', 'phase_dw'])
        dataframe.to_csv(
            path_or_buf=f'./predict/{predict_save_to}',
            index=False)
