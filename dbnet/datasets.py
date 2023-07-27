from io import BytesIO
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import lmdb
from PIL import Image
from torchvision.transforms import functional as TF


def image_to_bytes(image: Image.Image) -> bytes:
    """Save a PIL Image to a bytes buffer.

    Args:
        image (PIL.Image.Image):
            The PIL Image object to be converted.

    Returns:
        bs (bytes):
            The bytes representing the image in JPEG format with quality set to 100.
    """
    io = BytesIO()
    image.save(io, "WEBP", lossless=True)
    bs = io.getvalue()
    return bs


def bytes_to_image(bs: bytes) -> Image.Image:
    """Save a PIL Image to a bytes buffer.

    Args:
        image (PIL.Image.Image):
            The PIL Image object to be converted.

    Returns:
        bs (bytes):
            The bytes representing the image in JPEG format with quality set to 100.
    """
    io = BytesIO(bs)
    image = Image.open(io)
    return image


def numpy_to_bytes(img: np.ndarray) -> bytes:
    """Convert numpy -> PIL image -> bytes. Reverse of `bytes_to_numpy`.

    Args:
        img (np.ndarray):
            The input image as a NumPy array of shape [H, W] or [H, W, C].

    Returns:
        img_bin (bytes):
            The bytes object representing the image file.
    """
    img = (img * 255).astype("uint8")
    img = Image.open(img)
    img_bin = image_to_bytes(img)
    return img_bin


def bytes_to_numpy(image_bin: bytes) -> bytes:
    """Convert bytes -> PIL Image -> numpy. Reverse of `numpy_to_bytes`.

    Args:
        image_bin (bytes):
            The bytes object representing the image file.

    Returns:
        image (np.ndarray):
            The image as numpy array, shape [H, W, C] or [H, W] depending on the image.
            The values are float32 and normalized to 0, 1.
    """
    image = bytes_to_image(image_bin)
    image = np.array(image, "float32") / 255
    return image


def to_lmdb(
    image: Image.Image,
    proba_maps: np.ndarray,
    thresh_maps: np.ndarray,
    idx: int,
) -> Dict[bytes, bytes]:
    """
    Convert the DBNetSample to a dictionary suitable for storing in an LMDB database.

    Args:
        image (Image.Image):
            The input image.
        proba_maps (np.ndarray):
            The probability maps of shape [C, H, W].
        thresh_maps (np.ndarray):
            The threshold maps of shape [C, H, W].
        idx (int):
            The index of the sample.

    Returns:
        sample (dict):
            A dictionary representing the DBNetSample bytes keys and values.
            The dictionary is suitable for storing in an LMDB database.
            Keys: `image_{idx}`, `proba_maps_{idx}_{c}`, `thresh_maps_{idx}_{c}`,
            where `c` is the class index.
            The maps are stored as image file buffers for efficiency.
    """
    image_bin = image_to_bytes(image.convert("RGB"))
    proba_maps = [numpy_to_bytes(img) for img in proba_maps]
    thresh_maps = [numpy_to_bytes(img) for img in thresh_maps]

    sample = {f"image_{idx}": image_bin}
    for i in range(len(proba_maps)):
        pmap = proba_maps[i]
        tmap = thresh_maps[i]
        sample[f"proba_{idx}_{i}"] = pmap
        sample[f"thresh_{idx}_{i}"] = tmap
    return sample


class DBNetDataset(Dataset):
    """
    Dataset class for DBNet samples stored in an LMDB database.

    Args:
        root (str):
            The root directory of the LMDB database.

    Attributes:
        num_samples (int):
            The total number of samples in the dataset.
        num_classes (int):
            The number of classes (channels) in the probability and threshold maps.
        env (lmdb.Environment):
            The LMDB environment used to access the database.

    Methods:
        __len__():
            Get the number of samples in the dataset.
        __getitem__(idx):
            Get a sample from the dataset at the specified index.
            The sample includes the original image and its associated probability maps and threshold maps.
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        env = lmdb.open(root, readonly=True, lock=False)
        with env.begin() as txn:
            self.num_samples = int(txn.get("__len__".encode()))
            self.num_classes = int(txn.get("__num_classes__".encode()))
        self.env = env
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int):
                The index of the sample to retrieve.
            transform (Callable[T, T]):
                A transform function that takes `image`, `proba_maps`, and `threshold_maps`
                and returns the same. This is used for augmentations.

        Returns:
            image (torch.Tensor):
                A tensor in [3, H, W] that stores the input image.
            proba_maps (torch.Tensor):
                A tensor in [C, H, W] that stores the probability maps.
            thresh_maps (torch.Tensor):
                A tensor in [C, H, W] that stores the thresholds maps.
        """
        # Load from LMDB
        with self.env.begin() as txn:
            image = txn.get(f"image_{idx}".encode())
            pmap_bins = [
                txn.get(f"proba_{idx}_{c}".encode()) for c in range(self.num_classes)
            ]
            tmap_bins = [
                txn.get(f"threshold_{idx}_{c}".encode())
                for c in range(self.num_classes)
            ]

        # Load to tensor
        image = bytes_to_numpy(image)
        pmaps = [bytes_to_numpy(img) for img in pmap_bins]
        tmaps = [bytes_to_numpy(img) for img in tmap_bins]
        pmaps = np.stack(pmaps, axis=0)
        tmaps = np.stack(tmaps, axis=0)

        # Transformation if any
        if self.transform is not None:
            image, pmaps, tmaps = self.transform(image, pmaps, tmaps)

        # image: H W C -> C H W
        # stack proba maps and threshold maps
        image = np.transpose(image, (2, 0, 1))
        pmaps = np.stack(pmaps, dim=0)
        tmaps = np.stack(tmaps, dim=0)
        return image, pmaps, tmaps

    def visualize(self):
        from matplotlib import pyplot as plt

        for i in range(self):
            image, pmaps, tmaps = self[i]
            print(f"sample {i}")
            fig = self.visualize_sample(image, pmaps, tmaps)
            plt.show()

    @staticmethod
    def visualize_sample(
        image: np.ndarray,
        proba_maps: np.ndarray,
        thresh_maps: np.ndarray,
    ):
        """
        Visualize a sample from the DBNet dataset.

        This function creates a matplotlib figure to display the original image along with its
        associated probability maps and threshold maps.

        Args:
            image (np.ndarray):
                The NumPy array representing the original image, shape [H, W, 3].
            proba_maps (np.ndarray):
                The NumPy array containing probability maps generated by DBNet for the image.
                Shape [C, H, W].
            thresh_maps (np.ndarray):
                The NumPy array containing threshold maps generated by DBNet for the image.
                Shape [C, H, W].

        Returns:
            matplotlib.figure.Figure:
                The matplotlib figure containing the visualization of the sample.
        """
        from matplotlib import pyplot as plt

        # Create figure
        n = len(proba_maps)
        plot_count = len(proba_maps) * 2 + 1
        fig = plt.figure(figsize=(15, 15))

        # Draw image
        plt.subplot(1, plot_count, 1).set_axis_off()
        plt.imshow(image)
        plt.title(f"image")

        # Draw proba maps
        count = 2
        for i in range(n):
            plt.subplot(1, plot_count, count).set_axis_off()
            plt.imshow(proba_maps[i], cmap="gray")
            plt.title(f"proba map {i}")
            count += 1

        # Draw threshhold maps
        for i in range(n):
            plt.subplot(1, plot_count, count).set_axis_off()
            plt.imshow(thresh_maps[i], cmap="gray")
            plt.title(f"threshold map {i}")
            count += 1

        return fig
