import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def saveImages(epoch: int, images: np.ndarray, save_path: Path, n_rows: int, n_cols: int,
               fig_dim=(10, 10), name_suff: str = ""):
    assert len(images) == (n_rows * n_cols)

    if not save_path.is_dir():
        save_path.mkdir(parents=True)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_dim)
    ax = ax.flatten()

    for i in range(len(ax)):
        image = images[i]
        ax[i].imshow(image)
        ax[i].axis('off')

    fig.set_tight_layout(True)
    save_path = save_path / f"{str(epoch).zfill(6)}{name_suff}.jpg"
    fig.savefig(str(save_path))
    plt.close()


def writeImagesOnDisk(images: np.ndarray, img_folder_path: str = None) -> tuple:
    if img_folder_path is None:
        img_folder_path = Path(tempfile.mkdtemp())

    img_list = []
    for i in range(len(images)):
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=str(img_folder_path)).name
        plt.imsave(img_path, images[i])
        img_list.append(img_path)

    return img_folder_path, img_list


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
    brightness = x + magnitude
    return brightness


def randomSaturation(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    saturation = (x - x_mean) * magnitude + x_mean
    return saturation


def randomContrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    contrast = (x - x_mean) * magnitude + x_mean
    return contrast


def randomTranslation(x, ratio=0.125):
    batch = tf.shape(x)[0]
    img_size = tf.shape(x)[1:3]
    shift = tf.cast(tf.cast(img_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform([batch, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([batch, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(img_size[0], dtype=tf.int32), 0) + translation_x + 1, 0,
                              img_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(img_size[1], dtype=tf.int32), 0) + translation_y + 1, 0,
                              img_size[1] + 1)
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
    translation = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]),
                               tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return translation


def randomCutout(x, ratio=0.5):
    batch = tf.shape(x)[0]
    img_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(img_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=img_size[0] + (1 - cutout_size[0] % 2),
                                 dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=img_size[1] + (1 - cutout_size[1] % 2),
                                 dtype=tf.int32)
    batch_grid, grid_x, grid_y = tf.meshgrid(tf.range(batch, dtype=tf.int32),
                                             tf.range(cutout_size[0], dtype=tf.int32),
                                             tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack(
        [batch_grid, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch, img_size[0], img_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch, cutout_size[0], cutout_size[1]], dtype=tf.float32),
                      mask_shape), 0)
    cutout = x * tf.expand_dims(mask, axis=3)
    return cutout


augment_functions = {
    'color': [randomBrightness, randomSaturation, randomContrast],
    'translation': [randomTranslation],
    'cutout': [randomCutout],
}
