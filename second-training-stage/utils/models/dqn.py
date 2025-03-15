import torch
import torchvision.models as models
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self) -> None:
        super(DQN, self).__init__()

        # Extract features with pre-trained VGG16
        # self.ecc_features = load_eccNet(
        #     (
        #         1,
        #         3,
        #         64 * (16 * 2 - 1),
        #         64 * (16 * 2 - 1),
        #     )
        # )
        # vgg16 = models.vgg16(pretrained=True)
        # self.features = vgg16.features

        # Define convolutional layers for search features
        self.search = nn.Sequential(
            # nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Define convolutional layers for target features
        self.target = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define convolutional layers for concatenated features
        self.conv = nn.Conv2d(64*5, 64, kernel_size=3, stride=1, padding=1)

        # MLP
        self.fc1 = nn.Linear(64*8*8+4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

    def forward(self, input1, input2, input3, input4, input5, input6):
        # x1 = self.ecc_features(input1)
        # x2 = self.features(input2)
        # x3 = self.features(input3)
        # x4 = self.features(input4)
        # x5 = self.features(input5)

        x1 = self.search(input1)
        x2 = self.target(input2)
        x3 = self.target(input3)
        x4 = self.target(input4)
        x5 = self.target(input5)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = torch.relu(self.conv(x))
        x = x.view(-1, 64*8*8)
        x = torch.cat((x, input6), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)

        return output
    
if __name__=='__main__':
    device = torch.device("cuda")
    dqn = DQN().to(device)
    # for param in dqn.features.parameters():
    #     param.requires_grad = False
    # for param in dqn.ecc_features.parameters():
    #     param.requires_grad = False
    param_num = sum(p.numel() for p in dqn.parameters() if p.requires_grad)
    print(param_num)

    observation = torch.rand((1, 3, 1024, 1024), device=device)
    size = 16
    fixation = [8, 8]
    target_size = 64
    paddings = (
        (size - fixation[0] - 1) * target_size,
        fixation[0] * target_size,
        (size - fixation[1] - 1) * target_size,
        fixation[1] * target_size,
    )
    observation = torch.nn.functional.pad(
        observation, paddings, "constant", 0.5)
    target = []
    for t in range(4):
        target.append(torch.rand((1, 3, 256, 256), device=device))
    points = torch.rand((1, 4), device=device)

    q_value = dqn(observation, target[0], target[1], target[2], target[3], points)
    print(q_value.shape)