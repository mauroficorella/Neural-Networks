from functools import partial
import numpy as np
import tensorflow as tf
from scipy import linalg


@tf.function
def getImages(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image


def createTempDataset(paths: list, batch: int, image_height: int = None, image_width: int = None):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(getImages)
    if image_width is not None and image_height is not None:
        dataset = dataset.map(partial(tf.image.resize, size=(image_height, image_width)))
    dataset = dataset.batch(batch_size=batch)
    return dataset


def getEncodings(model: tf.keras.models.Model, dataset: tf.data.Dataset, max_nb_images: int):
    image_encodings_2048 = np.zeros((max_nb_images, 2048))

    for i, image_batch in enumerate(dataset):
        batch_size = np.shape(image_batch)[0]
        encodings = model(image_batch)

        start_index = i * batch_size
        end_index = start_index + batch_size
        image_encodings_2048[start_index:end_index] = encodings

    return image_encodings_2048


def getEncodingStats(encodings):
    mu = np.mean(encodings, axis=0)
    sigma = np.cov(encodings, rowvar=False)
    return mu, sigma


def getFidScore(real_mu, real_sigma, fake_mu, fake_sigma):  # from mu and sigma
    ssdiff = np.sum((real_mu - fake_mu) ** 2.0)
    covmean = linalg.sqrtm(real_sigma.dot(fake_sigma))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = ssdiff + np.trace(real_sigma) + np.trace(fake_sigma) - 2 * np.trace(covmean)
    return fid_score


class InceptionModel(tf.keras.models.Model):
    def __init__(self, height: int = None, width: int = None):
        super().__init__()
        self.model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                    weights="imagenet",
                                                                    input_shape=(height, width, 3),
                                                                    pooling="avg")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=False)


def calculateFid(inception_model,
                 real_paths: list,
                 fake_paths: list,
                 batch_size: int = 1,
                 image_height: int = None,
                 image_width: int = None):
    # Just to make sure we have the same number of images from both category
    nb_of_images = min(len(real_paths), len(fake_paths))
    real_paths = real_paths[:nb_of_images]
    fake_paths = fake_paths[:nb_of_images]

    read_dataset = createTempDataset(real_paths, batch_size, image_height, image_width)
    real_encodings = getEncodings(inception_model, read_dataset, nb_of_images)
    real_mu, real_sigma = getEncodingStats(real_encodings)

    fake_dataset = createTempDataset(fake_paths, batch_size, image_height, image_width)
    fake_encodings = getEncodings(inception_model, fake_dataset, nb_of_images)
    fake_mu, fake_sigma = getEncodingStats(fake_encodings)

    fid_score = getFidScore(real_mu, real_sigma, fake_mu, fake_sigma)

    return fid_score
