from torch import nn

class CNNTraffic(nn.Module):
    def __init__(self,input_shape:int,output_shape:int):
        super().__init__()

        self.layer1 = nn.Sequential(
          nn.Conv2d(in_channels = input_shape, out_channels = 32, kernel_size=5, stride=1, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=5, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2),
          nn.Dropout(p=.25)
          )

        self.layer2 = nn.Sequential(
          nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2),
          nn.Dropout(p=.25))


        self.fc = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=3136,out_features=256),
          nn.Dropout(p=.5),
          nn.Linear(in_features=256,out_features=output_shape))


    def forward(self,x):
   # print('error BEFORE LAYER 1 HERE')
  #   x = self.layer1(x)
  # #  print('error BEFORE LAYER 2 HERE')
  #   x = self.layer2(x)
  #  # print('error BEFORE LAYER 3 HERE')
  #   x = self.layer3(x)
   # print('error BEFORE CLASS LAYER')
        return self.fc(self.layer2(self.layer1(x)))
