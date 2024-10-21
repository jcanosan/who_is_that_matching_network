from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset
from torchvision import transforms
import torchvision.datasets

from matching_networks import MatchingNetworks


class ImageLoader:
    def __init__(self):
        """Collects the specified images from the specified dataset"""

        # The dataset may be changed to another, but it need to match with the
        # one in dialogue.py when running both scripts
        self._dataset_name = "sota_dataset"

        # Collect the paths of the images to be used in this section
        # The number of shots may be changed to have more images per category
        # The number of categories (n_labels) may also be changed to limit the
        # number of categories that will be taken from the system
        # n_labels = None -> it will take all the categories
        self._collect_local_imgs(k_shot=5, n_labels=None)

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    def _collect_local_imgs(
        self,
        k_shot=5,
        exclude_test_img_per_label=0,
        n_labels=None,
        file_extension=".png",
        random_imgs=False,
        random_labs=False
    ):
        """Collect the paths of the images for the support and the test sets and
        the labels.

        :param k_shot: number of images to take per label for training dataset.
        :param exclude_test_img_per_label: images to exclude from the test split.
        :param n_labels: number of labels.
        :param file_extension: extension of the image files in the dataset.
        :param random_imgs: if True, the images are randomized.
        :param random_labs: if True, the labels are randomized. Only performed if
        n_labels is not None.

        :return: List of images for support and test and the list of labels."""

        data_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(256),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset_path = f"utils/datasets/{self._dataset_name}/"
        self.dataset = torchvision.datasets.ImageFolder(
            root=dataset_path, transform=data_transform
        )
        labels = [img[1] for img in self.dataset]
        train_indices, test_indices, _, _ = train_test_split(
            range(len(labels)), labels, stratify=labels, test_size=0.75
        )
        train_subset = Subset(self.dataset, train_indices)
        test_subset = Subset(self.dataset, test_indices)
        self._train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=1, shuffle=True
        )
        self._test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=1, shuffle=True
        )


image_loader = ImageLoader()
matching_networks = MatchingNetworks(
    image_loader.train_loader, image_loader.test_loader
)
train_accuracy, losses = matching_networks.train(num_epochs=3)
print("train_accuracy", train_accuracy)
print("losses", losses)
