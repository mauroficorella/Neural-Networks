import shutil
from pathlib import Path
import tensorflow as tf
from dataset_utils import cropCenterImages, getInputNoise, denormalizeImages
from GAN import discrRealFakeLoss, discrReconstructionLoss, generatorLoss
from utils import writeImagesOnDisk, differentialAugmentation
from FID_utils import calculateFID


@tf.function
def train(G, D, optim_G, optim_D, images, diff_augm_policies: str = None) -> tuple:
    batch = tf.shape(images)[0]

    # Images for the I_{part} reconstruction loss
    imgs_batch_center_crop_128 = cropCenterImages(images, 128)

    # Images for the I reconstruction loss
    img_batch_128 = tf.image.resize(images, (128, 128))

    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
        noise_input = getInputNoise(batch)
        generated_imgs = G(noise_input, training=True)

        real_fake_out_logits_real_imgs, decoded_real_img, decoded_real_img_central_crop = D(
            differentialAugmentation(images, policy=diff_augm_policies), training=True)
        real_fake_out_logits_fake_imgs, _, _ = D(
            differentialAugmentation(generated_imgs, policy=diff_augm_policies), training=True)

        # Discriminator loss
        real_fake_loss_D = discrRealFakeLoss(
            real_fake_out_logits_real_imgs=real_fake_out_logits_real_imgs,
            real_fake_out_logits_fake_imgs=real_fake_out_logits_fake_imgs)
        I_reconstruction_loss_D = discrReconstructionLoss(real_img=img_batch_128, decoded_img=decoded_real_img)
        I_part_reconstruction_loss_D = discrReconstructionLoss(
            real_img=imgs_batch_center_crop_128,
            decoded_img=decoded_real_img_central_crop)
        loss_D = real_fake_loss_D + I_reconstruction_loss_D + I_part_reconstruction_loss_D

        # Generator loss
        loss_G = generatorLoss(real_fake_out_logits_fake_imgs=real_fake_out_logits_fake_imgs)

    gradients_G = tape_G.gradient(loss_G, G.trainable_variables)
    gradients_D = tape_D.gradient(loss_D, D.trainable_variables)

    optim_D.apply_gradients(zip(gradients_D, D.trainable_variables))
    optim_G.apply_gradients(zip(gradients_G, G.trainable_variables))

    return loss_G, loss_D, real_fake_loss_D, I_reconstruction_loss_D, I_part_reconstruction_loss_D


def evaluate(inception_model: tf.keras.models.Model,
             dataset: tf.data.Dataset,
             G: tf.keras.models.Model,
             batch: int,
             img_height: int,
             img_width: int,
             num_imgs: int = 128) -> float:
    num_batch = num_imgs // batch

    # Write real images from the dataset to the disk
    real_imgs_paths = []
    for x in dataset.take(num_batch):
        real_imgs = denormalizeImages(x, dtype=tf.uint8).numpy()
        _, real_imgs_file_paths = writeImagesOnDisk(real_imgs, img_folder_path=None)
        real_imgs_paths.extend(real_imgs_file_paths)

    # Generate images and write to the disk
    fake_imgs_paths = []
    for i in range(num_batch):
        input_noise = getInputNoise(batch)
        fake_imgs = G(input_noise)
        fake_imgs = denormalizeImages(fake_imgs, dtype=tf.uint8).numpy()
        _, fake_imgs_file_paths = writeImagesOnDisk(fake_imgs, img_folder_path=None)
        fake_imgs_paths.extend(fake_imgs_file_paths)

    FID_score = calculateFID(inception_model,
                             real_imgs_paths,
                             fake_imgs_paths,
                             batch=batch,
                             img_height=img_height,
                             img_width=img_width)

    # Cleanup, remove the folders with all the written files
    shutil.rmtree(Path(real_imgs_paths[0]).parent)
    shutil.rmtree(Path(fake_imgs_paths[0]).parent)

    return FID_score
