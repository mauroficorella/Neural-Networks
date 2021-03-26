from functools import partial
import numpy as np
import tensorflow as tf
from scipy import linalg


def getEncodings(model: tf.keras.models.Model, dataset: tf.data.Dataset, max_num_imgs: int):
    enc_img_2048 = np.zeros((max_num_imgs, 2048))

    for i, batch_img in enumerate(dataset):
        batch = np.shape(batch_img)[0]
        encodings = model(batch_img)

        start_index = i * batch
        end_index = start_index + batch
        enc_img_2048[start_index:end_index] = encodings

    return enc_img_2048


def getEncodingStats(encodings):
    mu = np.mean(encodings, axis=0)
    sigma = np.cov(encodings, rowvar=False)
    return mu, sigma


@tf.function
def getImages(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image


def createTempDataset(paths: list, batch: int, img_height: int = None, img_width: int = None):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(getImages)
    if img_width is not None and img_height is not None:
        dataset = dataset.map(partial(tf.image.resize, size=(img_height, img_width)))
    dataset = dataset.batch(batch_size=batch)
    return dataset


def getFIDScore(mu_real, sigma_real, mu_fake, sigma_fake):  # from mu and sigma
    # calculate sum squared difference between means
    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    FID_score = ssdiff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    return FID_score


def calculateFID(inception_model,
                 real_paths: list,
                 fake_paths: list,
                 batch: int = 1,
                 img_height: int = None,
                 img_width: int = None):
    # Just to make sure we have the same number of images from both category
    imgs_num = min(len(real_paths), len(fake_paths))
    real_paths = real_paths[:imgs_num]
    fake_paths = fake_paths[:imgs_num]

    dataset_real = createTempDataset(real_paths, batch, img_height, img_width)
    real_encodings = getEncodings(inception_model, dataset_real, imgs_num)
    mu_real, sigma_real = getEncodingStats(real_encodings)

    dataset_fake = createTempDataset(fake_paths, batch, img_height, img_width)
    fake_encodings = getEncodings(inception_model, dataset_fake, imgs_num)
    mu_fake, sigma_fake = getEncodingStats(fake_encodings)

    FID_score = getFIDScore(mu_real, sigma_real, mu_fake, sigma_fake)

    return FID_score


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
