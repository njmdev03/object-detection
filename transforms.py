from torchvision import transforms as T
import torchvision.transforms.functional as F
from  torchvision.transforms import ColorJitter
import random

class ComposeDetection:
    """Compose transforms for object detection."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    """Convert PIL image to tensor."""
    def __call__(self, image, target):
        return F.to_tensor(image), target

class Resize:
    """Resize image and scale boxes accordingly."""
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, image, target):
        # Make sure image is a PIL Image
        if not hasattr(image, "size"):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        w_orig, h_orig = image.size  # correct: size is a property, not a function
        image = F.resize(image, self.size)

        boxes = target["boxes"].clone().float()
        boxes[:, [0,2]] = boxes[:, [0,2]] * self.size[1] / w_orig
        boxes[:, [1,3]] = boxes[:, [1,3]] * self.size[0] / h_orig
        target["boxes"] = boxes

        return image, target

class RandomHorizontalFlip:
    """Random horizontal flip."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            w, _ = image.size
            boxes = target["boxes"].clone()
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
            target["boxes"] = boxes
        return image, target

class ColorJitterDetection:
    """Wrap torchvision ColorJitter to support (image, target) tuples."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.jitter = T.ColorJitter(brightness=brightness,
                                    contrast=contrast,
                                    saturation=saturation,
                                    hue=hue)

    def __call__(self, image, target):
        image = self.jitter(image)
        return image, target

def get_detection_transforms(train=True):
    transforms = [Resize((512,512))]

    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(ColorJitterDetection(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))

    transforms.append(ToTensor())
    return ComposeDetection(transforms)