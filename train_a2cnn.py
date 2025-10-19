# train_a2cnn.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from helper_lib import set_seed, get_device, save_checkpoint
from helper_lib.model import get_model
from helper_lib.trainer import fit

def main():
    set_seed(42)
    device = get_device()
    print("Device:", device)

    # CIFAR-10: 32x32 RGB -> resize to 64x64 as assignment spec
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2)

    model = get_model("A2CNN", num_classes=10).to(device)

    # quick training; increase epochs later if you want better accuracy
    history = fit(model, train_loader, val_loader=test_loader, epochs=5, lr=1e-3, device=device)

    # save weights
    save_checkpoint(model, "checkpoints/a2cnn_cifar10.pt")
    print("Saved to checkpoints/a2cnn_cifar10.pt")

if __name__ == "__main__":
    main()
