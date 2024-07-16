import random
from torchvision.transforms import v2

def apply_augmentation(input):
    # apply random augmentation
    # remember type and parameters
    # return transformed image
    type = random.randint(0,8)
    angle = random.randrange(0, 360)

    if type == 0:
        transform = v2.ColorJitter(brightness=(0, 2))
        transformed = transform(input)
    elif type == 1:
        transform = v2.ColorJitter(hue=(0, 0.15))
        transformed = transform(input)
    elif type == 2:
        transform = v2.ColorJitter(saturation=(0, 2))
        transformed = transform(input)
    elif type == 3:
        transform = v2.ColorJitter(contrast=(0, 2))
        transformed = transform(input)
    elif type == 4:
        transformed = v2.functional.horizontal_flip(input)
        return transformed, type
    elif type == 5:
        transformed = v2.functional.vertical_flip(input)
        return transformed, type
    elif type == 6:
        transformed = v2.functional.rotate(input,angle)
        return transformed, type, angle
    else:
        transform = v2.GaussianBlur(kernel_size=(3, 3), sigma=(1., 9.))
        transformed = transform(input)
        return transformed, type

    return transformed, type