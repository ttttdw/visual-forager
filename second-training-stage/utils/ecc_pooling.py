import torch
import torch.nn as nn
import math
from skimage.draw import disk


class EccPool(nn.Module):
    def __init__(
        self,
        rf_min=2,
        ecc_slope=0.2,
        deg2px=15,
        fovea_size=4,
        rf_quant=1,
        stride=2,
        pool_type="max",
        input_shape=None,
    ):
        super(EccPool, self).__init__()

        self.stride = stride
        self.pool_type = pool_type
        self.fovea_size = fovea_size
        self.rf_min = rf_min
        self.ecc_slope = ecc_slope
        self.deg2px = deg2px
        self.rf_quant = rf_quant
        self.base = input_shape[2]
        self.out_shape = int(((self.base + 1) / self.stride))
        self.channels = input_shape[0]
        self.device = torch.device("cuda")

        c = int(self.out_shape / 2)

        mask_size = torch.zeros(
            (self.channels, self.out_shape, self.out_shape))

        self.pools = []
        self.masks = []

        ecc = round((self.fovea_size * self.deg2px) / 2)
        if ecc > self.out_shape / 2:
            temp_mask = torch.ones(mask_size.shape)
            self.masks.append(temp_mask.to(self.device))
            self.pools.append(
                nn.AvgPool2d(
                    self.rf_min, self.stride, padding=(
                        self.rf_min - 1) // self.stride
                )
            )
        else:
            # first mask
            temp_mask = torch.clone(mask_size)
            rr, cc = disk((c, c), ecc)
            temp_mask[:, rr, cc] = 1
            self.masks.append(temp_mask.to(self.device))
            self.pools.append(
                nn.AvgPool2d(
                    self.rf_min, self.stride, padding=(
                        self.rf_min - 1) // self.stride
                )
            )

            # mid mask
            ecc += round((self.rf_quant * self.deg2px) / 2)
            while ecc < self.out_shape / 2:
                temp_mask = torch.clone(mask_size)
                rr, cc = disk((c, c), ecc)
                temp_mask[:, rr, cc] = 1
                rr, cc = disk(
                    (c, c), ecc - round((self.rf_quant * self.deg2px) / 2))
                temp_mask[:, rr, cc] = 0
                self.masks.append(temp_mask.to(self.device))
                rf_size = self.rf_min + round(
                    self.ecc_slope * (ecc * 2 - self.fovea_size * self.deg2px)
                )
                self.pools.append(
                    nn.AvgPool2d(rf_size, self.stride, padding=(
                        rf_size - 1) // self.stride)
                )
                ecc += round((self.rf_quant * self.deg2px) / 2)

            # end mask
            temp_mask = torch.ones(mask_size.shape)
            rr, cc = disk(
                (c, c), ecc - round((self.rf_quant * self.deg2px) / 2))
            temp_mask[:, rr, cc] = 0
            self.masks.append(temp_mask.to(self.device))
            rf_size = self.rf_min + round(
                self.ecc_slope * (ecc * 2 - self.fovea_size * self.deg2px)
            )
            self.pools.append(
                nn.AvgPool2d(rf_size, self.stride, padding=(
                    rf_size - 1) // self.stride)
            )

    def forward(self, input):
        out = torch.zeros((self.channels, self.out_shape,
                          self.out_shape), device=self.device)
        for mask, pool in zip(self.masks, self.pools):
            temp_pool = pool(input)
            out = mask * temp_pool + out
        return out


if __name__ == "__main__":
    model = EccPool()
    print(model)
