from functools import partial
import tensorflow as tf


def resizeAndNormalizeImages(images, resolution: int):  # preprocess
    """
    Resize and normalize the images tot he range [-1, 1]
    Args:
        images: batch of images (B, H, W, C)

    Returns:
        resized and normalized images
    """

    images = tf.image.resize(images, (resolution, resolution))
    images = tf.cast(images, tf.float32) - 127.5
    images = images / 127.5
    return images


def getImageFromPath(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return image


def getDataset(batch: int,
               dataset_folder: str,
               resolution: int,
               img_extension: str,
               flip_augment: bool = True,
               size_buff_shuffle: int = 100):
    dataset = tf.data.Dataset.list_files(dataset_folder + f"/*.{img_extension}")
    dataset = dataset.map(getImageFromPath)
    if flip_augment:
        dataset = dataset.map(tf.image.flip_left_right)
    dataset = dataset.map(partial(resizeAndNormalizeImages, resolution=resolution))
    dataset = dataset.shuffle(buffer_size=size_buff_shuffle).batch(batch).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def denormalizeImages(images, dtype=tf.float32):  # postprocess
    """
    De-Normalize the images to the range [0, 255]
    Args:
        images: batch of normalized images
        dtype: target dtype

    Returns:
        de-normalized images
    """

    images = (images * 127.5) + 127.5
    images = tf.cast(images, dtype)
    return images


def cropCenterImages(images, crop_target_resolution: int):
    """
    Crops the center of the images
    Args:
        images: shape: (B, H, W, 3), H should be equal to W
        crop_target_resolution: target resolution for the crop

    Returns:
        cropped images which has the shape: (B, crop_resolution, crop_resolution, 3)
    """

    crop_target_resolution = tf.cast(crop_target_resolution, tf.float32)
    crop_resolution_half = crop_target_resolution / 2
    img_height = tf.cast(tf.shape(images)[1], tf.float32)
    img_center = img_height / 2

    from_ = int(img_center - crop_resolution_half)
    to_ = int(img_center + crop_resolution_half)

    return images[:, from_:to_, from_:to_, :]


def getTestImages(batch: int, dataset_folder: str, resolution: int, img_extension: str):
    dataset = getDataset(batch, str(dataset_folder), resolution=resolution, img_extension=img_extension,
                         flip_augment=False, size_buff_shuffle=1)
    for x in dataset.take(1):
        return x


def getInputNoise(batch_size: int):
    return tf.random.normal(shape=(batch_size, 1, 1, 256), mean=0.0, stddev=1.0, dtype=tf.float32)