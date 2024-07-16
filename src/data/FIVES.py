import random
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import h5py
from torchvision.transforms import v2


class Fives(Dataset):
    def __init__(self, file_path, t: str, transform=None):
        hf = h5py.File(file_path, "r")

        self.transform = transform

        if t in ["train", "val", "test", "cal"]:
            self.images = hf[t]["images"]
            self.segmentations = hf[t]["label"]
            self.id = hf[t]["id"]
        else:
            raise ValueError(f"Unknown test/train/val/cal specifier: {t}")

    def __getitem__(self, index):

        image = self.images[index]
        segmentation = self.segmentations[index]

        # normalise image
        image = (image - image.mean(axis=(0, 1))) / image.std(axis=(0, 1))

        # change shape from (size, size, 3) to (3, size, size)
        image = np.moveaxis(image, -1, 0)

        # Convert to torch tensor
        image = torch.from_numpy(image)
        segmentation = torch.from_numpy(segmentation)

        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        segmentation = segmentation.type(torch.LongTensor)

        if self.transform == None:
            pass
        else:
            if random.random() > 0.5:
                out = self.transform(image)
                image = out[0]
                type = out[1]
                if type == 4:
                    segmentation = v2.functional.horizontal_flip(segmentation)
                elif type == 5:
                    segmentation = v2.functional.vertical_flip(segmentation)
                elif type == 6:
                    pil_segmentation = Image.fromarray(segmentation.numpy(), mode='F')
                    segmentation = v2.functional.rotate(pil_segmentation, -out[2])
                    segmentation = torch.from_numpy(np.array(segmentation).copy())
        return image, segmentation

    def __len__(self):
        return self.images.shape[0]

    def get_id(self, i):
        return self.id[i]