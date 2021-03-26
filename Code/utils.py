import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def saveImages(epoch: int, images: np.ndarray, save_folder: Path, rows: int, cols: int,
               figsize=(10, 10), name_suffix: str = ""):
    assert len(images) == (rows * cols)

    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()

    for i in range(len(axs)):
        image = images[i]
        axs[i].imshow(image)
        axs[i].axis('off')

    fig.set_tight_layout(True)
    save_path = save_folder / f"{str(epoch).zfill(6)}{name_suffix}.jpg"
    fig.savefig(str(save_path))
    plt.close()


def writeImagesOnDisk(images: np.ndarray, folder: str = None) -> tuple:
    if folder is None:
        folder = Path(tempfile.mkdtemp())

    file_list = []
    for i in range(len(images)):
        file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=str(folder)).name
        plt.imsave(file_path, images[i])
        file_list.append(file_path)

    return folder, file_list


# DIFFERENTIAL AUGMENTATION COMPUTATIONS
def differentialAugmentation(x, policy: str = None, channels_first=False):
    if policy:
        if channels_first:
            x = tf.transpose(x, [0, 2, 3, 1])
        for p in policy.split(','):
            for f in augment_functions[p]:
                x = f(x)
        if channels_first:
            x = tf.transpose(x, [0, 3, 1, 2])
    return x


def randomBrightness(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5
    x = x + magnitude
    return x


def randomSaturation(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def randomContrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def randomTranslation(x, ratio=0.125):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0,
                              image_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0,
                              image_size[1] + 1)
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
    x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]),
                                  tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return x


def randomCutout(x, ratio=0.5):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2),
                                 dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2),
                                 dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32),
                                             tf.range(cutout_size[0], dtype=tf.int32),
                                             tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack(
        [grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(
        1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32),
                          mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x


augment_functions = {
    'color': [randomBrightness, randomSaturation, randomContrast],
    'translation': [randomTranslation],
    'cutout': [randomCutout],
}
