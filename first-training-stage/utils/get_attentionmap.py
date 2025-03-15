import math
import torch
from torchvision import transforms
from utils.ecc_pooling import EccPool
import torch.nn.functional as F


device = torch.device('cuda')


def visual_process(observation, model_stimuli, mean, std, pe):
    observation = observation.float().permute(2, 0, 1) / 255
    obs_img = transforms.ToPILImage()(observation)
    observation = transforms.Normalize(mean=mean, std=std)(observation)
    output_stimuli = model_stimuli(observation.to(device))
    output_stimuli = output_stimuli
    return output_stimuli, obs_img


def generate_similarity_map(output_stimuli, MMconv):
    out = MMconv(output_stimuli)
    out = out / torch.max(out)

    out_img = transforms.ToPILImage()(out)

    return out.detach(), out_img


def ecc_visual_process(
    observation, model_stimuli, fixation, mean, std, pe, size, target_size
):
    observation = observation.float().permute(2, 0, 1) / 255
    # obs_img = transforms.ToPILImage()(observation)
    paddings = (
        (size - fixation[0] - 1) * target_size,
        fixation[0] * target_size,
        (size - fixation[1] - 1) * target_size,
        fixation[1] * target_size,
    )
    observation = torch.nn.functional.pad(
        observation, paddings, "constant", 0.5)
    # ecc_pool = EccPool(input_shape=observation.shape)
    # ecc_out = ecc_pool(observation.to(device))
    # ecc_out = ecc_out[
    #     :,
    #     int(target_size / 2)
    #     * (size - fixation[1] - 1): int(target_size / 2)
    #     * (2 * size - fixation[1] - 1),
    #     int(target_size / 2)
    #     * (size - fixation[0] - 1): int(target_size / 2)
    #     * (2 * size - fixation[0] - 1),
    # ]
    ecc_img = None  # transforms.ToPILImage()(ecc_out)
    observation = transforms.Normalize(mean=mean, std=std)(observation)
    output_stimuli = model_stimuli(observation.to(device))
    output_stimuli = output_stimuli[
        :,
        2 * (size - fixation[1] - 1): 2 * (2 * size - fixation[1] - 1),
        2 * (size - fixation[0] - 1): 2 * (2 * size - fixation[0] - 1),
    ]
    output_stimuli = output_stimuli + pe

    return output_stimuli, None, ecc_img


def generate_ior_map(fixations, size):
    ior_map = torch.ones((1, size, size))
    for f in fixations[1:]:
        ior_map[0, f[1], f[0]] = 0
    ior_img = transforms.ToPILImage()(ior_map)
    return ior_map.to(device), ior_img


def generate_saccade_map(fixation, size):
    dva = 4
    la = 6
    saccade_map = torch.zeros((1, size, size))
    for i in range(size):
        for j in range(size):
            d = (
                round(math.sqrt((fixation[1] - i) **
                      2 + (fixation[0] - j) ** 2)) * dva
                + 1
            )
            p = la**d * math.exp(-la) / math.factorial(d)
            saccade_map[0, i, j] = p * 5
    saccade_img = transforms.ToPILImage()(saccade_map)
    return saccade_map.to(device), saccade_img


def get_attention_map(
    observation,
    fixations,
    model_stimuli,
    MMconvs,
    env_config,
    mean=0.6245,
    std=0.3713,
    pe=None,
):
    size = env_config["variable"]["size"]
    attention_maps = []
    output_stimuli, obs_img = visual_process(
        observation, model_stimuli, mean, std, pe)
    ior_map, ior_img = generate_ior_map(fixations, size)

    saccade_map, saccade_img = generate_saccade_map(fixations[-1], size)

    for MMconv in MMconvs:
        similarity_map, similarity_img = generate_similarity_map(
            output_stimuli, MMconv)
        attention_map = similarity_map + saccade_map * 0
        # attention_map = (attention_map - torch.min(attention_map)) / \
        #     (torch.max(attention_map) - torch.min(attention_map))

        # attention_map = F.tanh(attention_map)
        attention_map = attention_map * ior_map
        attention_map = (attention_map - attention_map.mean()) / \
            attention_map.std()
        # attention_img = transforms.ToPILImage()(attention_map)
        # attention_img = transforms.ToPILImage()(attention_map)
        # attention_img.save("attention_map.jpg")
        attention_maps.append(attention_map)

    # obs_img.save("obs_img.jpg")
    # ior_img.save("ior_map.jpg")
    # saccade_img.save("saccade_img.jpg")
    attention_map = torch.cat(attention_maps).unsqueeze(0)
    # attention_map = (attention_map - attention_map.mean()) / \
    #     attention_map.std()
    return attention_map


def get_eccattention_map(
    observation,
    fixations,
    model_stimuli,
    MMconvs,
    env_config,
    mean=0.6245,
    std=0.3713,
    pe=None,
):
    size = env_config["variable"]["size"]
    target_size = env_config["variable"]["target size"]
    attention_maps = []
    output_stimuli, obs_img, ecc_img = ecc_visual_process(
        observation, model_stimuli, fixations[-1], mean, std, pe, size, target_size
    )
    ior_map, ior_img = generate_ior_map(fixations, size)

    saccade_map, saccade_img = generate_saccade_map(fixations[-1], size)

    for MMconv in MMconvs:
        similarity_map, similarity_img = generate_similarity_map(
            output_stimuli, MMconv)
        # similarity_img.save("similarity_map.jpg")
        attention_map = similarity_map + saccade_map * 0
        attention_map = attention_map * ior_map
        attention_map = (attention_map - attention_map.mean()) / \
            attention_map.std()
        # attention_img = transforms.ToPILImage()(attention_map)
        # attention_img.save("similarity map.jpg")
        attention_maps.append(attention_map)
    # obs_img.save("obs_img.jpg")
    # ecc_img.save("ecc_img.jpg")
    # ior_img.save("ior_map.jpg")
    # saccade_img.save("saccade_img.jpg")
    attention_map = torch.cat(attention_maps).unsqueeze(0)

    return attention_map
