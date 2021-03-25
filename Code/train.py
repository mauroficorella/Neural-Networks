import shutil
from pathlib import Path
from data import *
from generator import Generator
from discriminator import Discriminator
from fid import InceptionModel
from training_routine import train_step, evaluation_step
from visualization import visualize_images_on_grid_and_save
import argparse
import sys
import signal

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="experiment", help="Name of the experiment")
parser.add_argument("--override", action="store_true", help="Removes previous experiment with same name")
parser.add_argument("--data-folder", type=str, required=True, help="Folder with the images")
parser.add_argument("--resolution", type=int, default=512, help="Either 256, 512 or 1024. Default is 512.")
parser.add_argument("--generator-weights", type=str, default=None)
parser.add_argument("--discriminator-weights", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--G-learning-rate", type=float, default=2e-4, help="Learning rate for the Generator")
parser.add_argument("--D-learning-rate", type=float, default=2e-4, help="Learning rate for the Discriminator")
parser.add_argument("--diff-augment", action="store_true", help="Apply diff augmentation")
parser.add_argument("--fid", action="store_true", help="If this is used, FID will be evaluated")
parser.add_argument("--fid-frequency", type=int, default=1, help="FID will be evaluated at this frequency (epochs)")
parser.add_argument("--fid-number-of-images", type=int,
                    default=128, help="This many images will be used for the FID calculation")

args = parser.parse_args()

print(args)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    epoch_file = open(checkpoints_folder / "epoch_file.txt", "w")
    epoch_file.write(str(epoch))
    epoch_file.close()
    sys.exit(0)


physical_devices = tf.config.list_physical_devices("GPU")
_ = [tf.config.experimental.set_memory_growth(x, True) for x in physical_devices]

results_folder = Path("Results") / args.name
checkpoints_folder = results_folder / "checkpoints"

if results_folder.is_dir():
    args.generator_weights = results_folder / "checkpoints/G_checkpoint.h5"
    args.discriminator_weights = results_folder / "checkpoints/D_checkpoint.h5"
    with open(checkpoints_folder / "epoch_file.txt") as f:
        start_epoch = int(f.read())


if not checkpoints_folder.is_dir():
    checkpoints_folder.mkdir(parents=True)
tensorboard_logs_folder = results_folder / "tensorboard_logs"
if not tensorboard_logs_folder.is_dir():
    tensorboard_logs_folder.mkdir(parents=True)

resolution = args.resolution
batch_size = args.batch_size
epochs = args.epochs
dataset_folder = args.data_folder

dataset = create_dataset(batch_size=batch_size, folder=dataset_folder, resolution=resolution,
                         use_flip_augmentation=True, shuffle_buffer_size=500)

G = Generator(resolution)
sample_G_output = G.initialize(batch_size)
if args.generator_weights is not None:
    G.built = True
    G.load_weights(args.generator_weights)
    print("Weights are loaded for G")
print(f"[Model G] output shape: {sample_G_output.shape}")

D = Discriminator(resolution)
sample_D_output = D.initialize(batch_size)
if args.discriminator_weights is not None:
    D.built = True
    D.load_weights(args.discriminator_weights)
    print("Weights are loaded for D")
else:
    start_epoch = 0
print(f"[Model D] real_fake output shape: {sample_D_output[0].shape}")
print(f"[Model D] Image output shape: {sample_D_output[1].shape}")
print(f"[Model D] Image part output shape: {sample_D_output[2].shape}")

G_optimizer = tf.optimizers.Adam(learning_rate=args.G_learning_rate)
D_optimizer = tf.optimizers.Adam(learning_rate=args.D_learning_rate)

if args.fid:
    # Model for the FID calculation
    fid_inception_model = InceptionModel(height=resolution, width=resolution)

test_input_size = 25
test_input_for_generation = create_input_noise(test_input_size)
test_images = get_test_images(test_input_size, dataset_folder, resolution)

tb_file_writer = tf.summary.create_file_writer(str(tensorboard_logs_folder))
tb_file_writer.set_as_default()

G_loss_metric = tf.keras.metrics.Mean()
D_loss_metric = tf.keras.metrics.Mean()
D_real_fake_loss_metric = tf.keras.metrics.Mean()
D_I_reconstruction_loss_metric = tf.keras.metrics.Mean()
D_I_part_reconstruction_loss_metric = tf.keras.metrics.Mean()

diff_augment_policies = None
if args.diff_augment:
    diff_augment_policies = "color,translation,cutout"

signal.signal(signal.SIGINT, signal_handler)

fid_score_best = 5000

for epoch in range(start_epoch, epochs):
    print(f"Epoch {epoch} -------------")
    for step, image_batch in enumerate(dataset):
        G_loss, D_loss, D_real_fake_loss, D_I_reconstruction_loss, D_I_part_reconstruction_loss = train_step(
            G=G,
            D=D,
            G_optimizer=G_optimizer,
            D_optimizer=D_optimizer,
            images=image_batch,
            diff_augmenter_policies=diff_augment_policies)

        G_loss_metric(G_loss)
        D_loss_metric(D_loss)
        D_real_fake_loss_metric(D_real_fake_loss)
        D_I_reconstruction_loss_metric(D_I_reconstruction_loss)
        D_I_part_reconstruction_loss_metric(D_I_part_reconstruction_loss)

        if step % 100 == 0 and step != 0:
            print(f"\tStep {step} - "
                  f"G loss value: {G_loss_metric.result():.4f} | "
                  f"D loss value: {D_loss_metric.result():.4f} | "
                  f"D real_fake loss value: {D_real_fake_loss_metric.result():.4f} | "
                  f"D I recon loss value: {D_I_reconstruction_loss_metric.result():.4f} | "
                  f"D I part recon loss value: {D_I_part_reconstruction_loss_metric.result():.4f}")

    if args.fid:
        if epoch % args.fid_frequency == 0:
            fid_score = evaluation_step(inception_model=fid_inception_model,
                                        dataset=dataset,
                                        G=G,
                                        batch_size=batch_size,
                                        image_height=resolution,
                                        image_width=resolution,
                                        nb_of_images_to_use=args.fid_number_of_images)
            print(f"[FID] {fid_score:.2f}")
            tf.summary.scalar("FID_score", fid_score, epoch)

    tf.summary.scalar("G_loss/G_loss", G_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_loss", D_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_real_fake_loss", D_real_fake_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_I_reconstruction_loss", D_I_reconstruction_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_I_part_reconstruction_loss", D_I_part_reconstruction_loss_metric.result(), epoch)

    print(f"Epoch number: {epoch} - "
          f"G loss value: {G_loss_metric.result():.4f} | "
          f"D loss value: {D_loss_metric.result():.4f} | "
          f"D real_fake loss value: {D_real_fake_loss_metric.result():.4f} | "
          f"D I recon loss: {D_I_reconstruction_loss_metric.result():.4f} | "
          f"D I part recon loss: {D_I_part_reconstruction_loss_metric.result():.4f}")

    G_loss_metric.reset_states()
    D_loss_metric.reset_states()
    D_real_fake_loss_metric.reset_states()
    D_I_part_reconstruction_loss_metric.reset_states()
    D_I_reconstruction_loss_metric.reset_states()

    if args.fid:
        if fid_score < fid_score_best: #TODO scrivere il fid in un file
            fid_score_best = fid_score
            G.save_weights(str(checkpoints_folder / "G_checkpoint.h5"))
            D.save_weights(str(checkpoints_folder / "D_checkpoint.h5"))

    # Generate test images
    generated_images = G(test_input_for_generation, training=False)
    generated_images = postprocess_images(generated_images, dtype=tf.uint8).numpy()
    visualize_images_on_grid_and_save(epoch, generated_images, results_folder / "generated_images", 5, 5)

    # Generate reconstructions from Discriminator
    _, decoded_images, decoded_part_images = D(test_images, training=False)
    decoded_images = postprocess_images(decoded_images, dtype=tf.uint8).numpy()
    decoded_part_images = postprocess_images(decoded_part_images, dtype=tf.uint8).numpy()
    visualize_images_on_grid_and_save(epoch, decoded_images, results_folder / "reconstructed_whole_images", 5, 5)
    visualize_images_on_grid_and_save(epoch, decoded_part_images,
                                      results_folder / "reconstructed_part_images", 5, 5)

    if epoch == epochs-1:
        print("ciao")
        file = open(checkpoints_folder / "epoch_file.txt", "w+")
        file.write(str(epoch))
        file.close()
