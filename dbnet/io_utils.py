from io import BytesIO
from typing import List, Dict, Any

import numpy as np
import lmdb
from PIL import Image


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


def sample_to_lmdb(
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


def dataset_to_lmdb(
    output_path: str,
    dataset: Any,
    num_classes: int,
    map_size: int = int(1e12),
    cache_size: int = 100,
):
    """
    Convert a dataset to an LMDB database.

    Args:
        output_path (str):
            The output path of the LMDB database.
        dataset (Any):
            The input dataset. The dataset should be iterable and return
            samples in the format (image, proba_maps, thresh_maps).
        num_classes (int):
            The number of classes (channels) in the probability and threshold maps.
        map_size (int, optional):
            The maximum size of the LMDB database. Default is 1 TB (10^12 bytes).
        cache_size (int, optional):
            The number of samples to write to the database at once. Default is 100.

    Returns:
        str:
            The path to the generated LMDB database.
    """
    num_samples = len(output_path)

    # Writing context
    env = lmdb.open(output_path, map_size=map_size)

    def write_cache(cache: List[Dict[bytes, bytes]]):
        with env.begin(write=True) as txn:
            for sample in cache:
                for k, v in sample.items():
                    txn.put(k, v)

    # Write dataset
    pbar = tqdm(range(num_samples), "Creating dataset")
    cache = []
    cache_count = 0
    for idx, targets in enumerate(data_gen):
        image, proba_maps, thresh_maps = targets
        assert len(proba_maps) == num_classes
        assert len(thresh_maps) == num_classes
        sample = sample_to_lmdb(image, proba_maps, thresh_maps, idx)
        cache.append(sample)
        cache_count = cache_count + 1
        if cache_count == cache_size:
            write_cache(cache)
            cache = []
            cache_count = 0
    write_cache(cache)

    # Write metadata
    metadata = dict()
    metadata["__len__".encode()] = str(num_samples).encode()
    metadata["__num_classes__".encode()] = str(num_classes).encode()
    write_cache(metadata)

    # Clean up and return the output path
    env.close()
    print(f"Output is written in {output_path}")
    return output_path
