import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
import os
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

DATASET_PATH = "./data/"
CHECKPOINT = "./saved_models/"

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


train_dataset = datasets.VOCSegmentation(
    "data", image_set="train", download=True, transform=train_transform
)

val_dataset = datasets.VOCSegmentation(
    "data", image_set="val", download=True, transform=ToTensor()
)

test_dataset = datasets.VOCSegmentation(
    "data", year="2007", image_set="test", download=True, transform=ToTensor()
)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

os.makedirs(CHECKPOINT, exist_ok=True)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
)

val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

if __name__ == "__main__":
    img = train_loader[0]
    # NUM_IMAGES = 4
    # images = [train_dataset[idx][0] for idx in range(NUM_IMAGES)]
    # orig_images = [
    #     Image.fromarray(train_dataset.images[idx]) for idx in range(NUM_IMAGES)
    # ]
    # orig_images = [img for img in orig_images]
    #
    # img_grid = torchvision.utils.make_grid(
    #     torch.stack(images + orig_images, dim=0), nrow=4, normalize=True, pad_value=0.5
    # )
    # img_grid = img_grid.permute(1, 2, 0)
    #
    # plt.figure(figsize=(8, 8))
    # plt.title("Augmentation examples on CIFAR10")
    # plt.imshow(img_grid)
    # plt.axis("off")
    # plt.show()
    # plt.close()
