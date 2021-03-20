import shutil
from pathlib import Path

import tensorflow as tf
from data import center_crop_images, create_input_noise, postprocess_images
from diff_augment import diff_augment
from losses import discriminator_real_fake_loss, discriminator_reconstruction_loss, generator_loss
from visualization import write_images_to_disk
from fid import calculate_FID




@tf.function
def train_step(G, D, G_optimizer, D_optimizer, images, diff_augmenter_policies: str = None) -> tuple:
    batch_size = tf.shape(images)[0]

    # Images for the I_{part} reconstruction loss
    images_batch_center_crop_128 = center_crop_images(images, 128)

    # Images for the I reconstruction loss
    image_batch_128 = tf.image.resize(images, (128, 128))

    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
        noise_input = create_input_noise(batch_size)
        generated_images = G(noise_input, training=True)

        real_fake_output_logits_on_real_images, decoded_real_image, decoded_real_image_central_crop = D(
            diff_augment(images, policy=diff_augmenter_policies), training=True)
        real_fake_output_logits_on_fake_images, _, _ = D(
            diff_augment(generated_images, policy=diff_augmenter_policies), training=True)

        # Discriminator loss
        D_real_fake_loss = discriminator_real_fake_loss(
            real_fake_output_logits_on_real_images=real_fake_output_logits_on_real_images,
            real_fake_output_logits_on_fake_images=real_fake_output_logits_on_fake_images)
        D_I_reconstruction_loss = discriminator_reconstruction_loss(real_image=image_batch_128,
                                                                            decoded_image=decoded_real_image)
        D_I_part_reconstruction_loss = discriminator_reconstruction_loss(
            real_image=images_batch_center_crop_128,
            decoded_image=decoded_real_image_central_crop)
        D_loss = D_real_fake_loss + D_I_reconstruction_loss + D_I_part_reconstruction_loss

        # Generator loss
        G_loss = generator_loss(real_fake_output_logits_on_fake_images=real_fake_output_logits_on_fake_images)

    G_gradients = tape_G.gradient(G_loss, G.trainable_variables)
    D_gradients = tape_D.gradient(D_loss, D.trainable_variables)

    D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables))
    G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))

    return G_loss, D_loss, D_real_fake_loss, D_I_reconstruction_loss, D_I_part_reconstruction_loss


def evaluation_step(inception_model:tf.keras.models.Model,
                    dataset: tf.data.Dataset,
                    G: tf.keras.models.Model,
                    batch_size: int,
                    image_height: int,
                    image_width: int,
                    nb_of_images_to_use: int = 128) -> float:
    number_of_batches = nb_of_images_to_use // batch_size

    # Write real images from the dataset to the disk
    real_paths = []
    for x in dataset.take(number_of_batches):
        real_images = postprocess_images(x, dtype=tf.uint8).numpy()
        _, real_images_file_paths = write_images_to_disk(real_images, folder=None)
        real_paths.extend(real_images_file_paths)

    # Generate images and write to the disk
    fake_paths = []
    for i in range(number_of_batches):
        input_noise = create_input_noise(batch_size)
        fake_images = G(input_noise)
        fake_images = postprocess_images(fake_images, dtype=tf.uint8).numpy()
        _, fake_images_file_paths = write_images_to_disk(fake_images, folder=None)
        fake_paths.extend(fake_images_file_paths)

    fid_score = calculate_FID(inception_model,
                                      real_paths,
                                      fake_paths,
                                      batch_size=batch_size,
                                      image_height=image_height,
                                      image_width=image_width)

    # Cleanup, remove the folders with all the written files
    shutil.rmtree(Path(real_paths[0]).parent)
    shutil.rmtree(Path(fake_paths[0]).parent)

    return fid_score
