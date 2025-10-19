from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir, batch_size=32, train=True):
    """
    Create a DataLoader for a given dataset directory.

    Args:
        data_dir (str): Path to dataset folder.
        batch_size (int): Number of samples per batch.
        train (bool): Whether to load training or test set.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root=data_dir + ('/train' if train else '/test'),
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
