import torch
import torch.nn as nn
import time


class ResNet(nn.Module):

    def _make_block(self, in_channels, out_channels, strid=1, first_block: bool = False):
        if first_block:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=strid),
                nn.BatchNorm2d(num_features=out_channels))

        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                          stride=strid),
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(num_features=out_channels),
            )

    def __init__(self, num_classes=10):
        super().__init__()
        # activation function
        self.relu_func = nn.ReLU()

        # max pool
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Convolution neural network
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.first_block = self._make_block(in_channels=64, out_channels=64, strid=1, first_block=True)
        self.stage1_b1 = self._make_block(in_channels=64, strid=1, out_channels=64)
        self.stage1_b2 = self._make_block(in_channels=64, strid=1, out_channels=64)

        self.second_block = self._make_block(in_channels=64, out_channels=128, strid=2, first_block=True)
        self.stage2_b1 = self._make_block(in_channels=64, strid=2, out_channels=128)
        self.stage2_b2 = self._make_block(in_channels=128, strid=1, out_channels=128)

        self.third_block = self._make_block(in_channels=128, out_channels=256, strid=2, first_block=True)
        self.stage3_b1 = self._make_block(in_channels=128, strid=2, out_channels=256)
        self.stage3_b2 = self._make_block(in_channels=256, strid=1, out_channels=256)

        self.forth_block = self._make_block(in_channels=256, out_channels=512, strid=2, first_block=True)
        self.stage4_b1 = self._make_block(in_channels=256, strid=2, out_channels=512)
        self.stage4_b2 = self._make_block(in_channels=512, strid=1, out_channels=512)

        # Fully connected neural network
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512 * 7 * 7, out_features=1024, bias=True),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_classes, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        # STEM
        x = self.stem(x)

        # STAGE 1
        identity = self.first_block(x)

        x = self.stage1_b1(x)
        x = self.stage1_b2(x)

        # Shortcut 1
        x += identity
        x = self.relu_func(x)

        # STAGE 2
        identity = self.second_block(x)

        x = self.stage2_b1(x)
        x = self.stage2_b2(x)

        # Shortcut 2
        x += identity
        x = self.relu_func(x)

        # STAGE 3
        identity = self.third_block(x)

        x = self.stage3_b1(x)
        x = self.stage3_b2(x)

        # Shortcut 3
        x += identity
        x = self.relu_func(x)

        # STAGE 4
        identity = self.forth_block(x)

        x = self.stage4_b1(x)
        x = self.stage4_b2(x)

        # Shortcut 4
        x += identity
        x = self.relu_func(x)

        # Flatten the matrix
        x = x.view(x.size(0), -1)  # x = torch.flatten(x, 1)

        # Sau khi su dung fully connected thi bo vao activation function
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    start = time.time()
    random_image = torch.rand(1, 3, 224, 224)  # [B,C,H,W]
    model = ResNet()  # Define a Convolutional neural network
    predictions = model(random_image)  # Forward pass
    print('Prediction shape:',predictions.shape)
    end = time.time()
    print(f"Time for 1 pic: {round(end - start, 2)}")
