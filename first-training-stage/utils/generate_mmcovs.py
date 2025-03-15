import sys
import torch
import os
import yaml
from torchvision import transforms
from PIL import Image


def generate_mmconvs(filenames, vgg_model):
    device = torch.device('cuda')

    model_target = vgg_model.features
    transform = transforms.Compose([transforms.PILToTensor()])
    with open("visual_foraging_gym/envs/env_config.yml", "r") as file:
        env_config = yaml.safe_load(file)
    target_size = env_config["variable"]["target size"]
    MMconvs, target_images = [], []
    dir = "visual_foraging_gym/envs/OBJECTSALL"
    for filename in filenames:
        target_image = Image.open(os.path.join(dir, filename))
        target_image = target_image.resize((target_size, target_size))
        target_image = transform(target_image).float() / 255
        target_images.append(target_image)
    # normalize targets
    mean, std, var = 0, 0, 0
    for target_image in target_images:
        mean += torch.mean(target_image)
        std += torch.std(target_image)
        var += torch.var(target_image)
    mean, std, var = (
        mean / len(target_images),
        std / len(target_images),
        var / len(target_images),
    )
    # define MMconvs
    for target_image in target_images:
        target_image = transforms.Normalize(mean=0.8087, std= 0.2568)(target_image)
        output_target = model_target(target_image.to(device))

        MMconv = torch.nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=output_target.shape[2],
            stride=2,
        )
        MMconv.weight.data = torch.nn.Parameter(output_target.unsqueeze(0))
        MMconvs.append(MMconv.to(device))
    return MMconvs, 0.8087, 0.2568