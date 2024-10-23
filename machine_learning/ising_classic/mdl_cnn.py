import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, lattice_length: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        nodes = int(32 * (lattice_length-4)/2 * (lattice_length-4)/2)
        # nodes = int(32 * (lattice_length-2) * (lattice_length-2))
        self.fc3 = nn.Linear(nodes, 100)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(100, 2)  # 512
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input 1 x L x L, output 32 x (L-2) x (L-2)
        x = self.conv1(x.float())
        x = self.act1(x)
        x = self.drop1(x)
        # print(x.shape)
        # input 32 x (L-2) x (L-2), output 32 x (L-4) x (L-4)
        x = self.act2(self.conv2(x))
        # input 32 x (L-4) x (L-4), output 32 x (L-4)/2 x (L-4)/2
        x = self.pool2(x)
        # input 32 x (L-4)/2 x (L-4)/2, output 32 * (L-4)/2 * (L-4)/2
        x = self.flat(x)
        # print(x.shape)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 2
        x = self.fc4(x)
        return self.softmax(x)
        # return x