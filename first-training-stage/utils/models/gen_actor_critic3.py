import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms
import yaml

## actor and critic share some weights
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        with open('visual_foraging_gym/envs/env_config.yml', 'r') as file:
            env_config = yaml.safe_load(file)
        size = env_config['variable']['size']
        
        self.conv1 = nn.Conv2d(4, 8, 1)
        self.conv2 = nn.Conv2d(8, 16, 1)
        self.conv3 = nn.Conv2d(16, 1, 1)

        self.conv4 = nn.Conv2d(4, 8, 9)
        self.conv5 = nn.Conv2d(8, 12, 5)
        self.conv6 = nn.Conv2d(12, 16, 3)

        self.fc4 = nn.Linear(size*size, 64)
        self.fc5 = nn.Linear(68+64, 64)
        self.fc6 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, attention_map, values):
        values = torch.log(values)
        values = f.softmax(values, 1)
        # values = f.relu(self.fc8(values))
        # values = f.relu(self.fc10(values))
        # values = f.relu(self.fc9(values))

        out = f.leaky_relu(self.conv4(attention_map))
        out = f.leaky_relu(self.conv5(out))
        out = f.leaky_relu(self.conv6(out))
        out = out.view(-1, self.num_flat_features(out))
        out1 = torch.cat((out, values), 1)

        values = values.view(-1, 4, 1, 1)
        attention_map = attention_map * values
        # forward
        out = f.relu(self.conv1(attention_map))
        out = f.relu(self.conv2(out))
        out = f.relu(self.conv3(out))
        # out_img = transforms.ToPILImage()(out.squeeze())
        # out_img.save('attention'+'.jpg')
        # torch.save({
        #     'attention map': out
        # }, 'attention.pt')
        policy = out.view(-1, self.num_flat_features(out))
        out = f.leaky_relu(self.fc4(policy))
        out = torch.cat((out, out1),1)
        out = self.dropout(f.leaky_relu(self.fc5(out)))
        value = self.dropout(f.leaky_relu(self.fc6(out)))
        # policy = self.dropout(f.relu(self.fc1(attention_map)))
        # policy1 = self.dropout(f.relu(self.fc2(policy)))
        # out = self.dropout(f.relu(self.fc21(out)))
        # out = self.dropout(f.relu(self.fc22(out)))
        # out = self.dropout(f.relu(self.fc23(out)))
        # policy = self.dropout(f.relu(self.fc3(policy1)))
        

        return policy, value

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    model = Actor()
    print(model)
    print(model.conv1.weight)
