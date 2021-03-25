from pathlib import Path
from data import *
from GAN import Generator, Discriminator
from fid import InceptionModel
from training_routine import train, evaluate
from visualization import saveImages
import argparse
import sys
import signal
import time
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="experiment", help="Name of the experiment")
parser.add_argument("--override", action="store_true", help="Removes previous experiment with same name")
parser.add_argument("--data-folder", type=str, required=True, help="Folder with the images")
parser.add_argument("--resolution", type=int, default=512, help="Either 256, 512 or 1024. Default is 512.")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--G-learning-rate", type=float, default=2e-4, help="Learning rate for the Generator")
parser.add_argument("--D-learning-rate", type=float, default=2e-4, help="Learning rate for the Discriminator")
parser.add_argument("--diff-augment", action="store_true", help="Apply diff augmentation")
parser.add_argument("--fid", action="store_true", help="If this is used, FID will be evaluated")
parser.add_argument("--fid-frequency", type=int, default=1, help="FID will be evaluated at this frequency (epochs)")
parser.add_argument("--fid-number-of-images", type=int,
                    default=128, help="This many images will be used for the FID calculation")
parser.add_argument("--img_extension", type=str, default="jpg",
                    help="Extension of the images used for the training (png, jpg, jpeg)")

args = parser.parse_args()

print(args)


def signalHandler(sig, frame):
    print('You pressed Ctrl+C!')
    epoch_file = open(checkpoints_folder / "epoch_file.txt", "w")
    epoch_file.write(str(epoch))
    epoch_file.close()
    sys.exit(0)


physical_devices = tf.config.list_physical_devices("GPU")
_ = [tf.config.experimental.set_memory_growth(x, True) for x in physical_devices]

results_folder = Path("Results") / args.name
checkpoints_folder = results_folder / "checkpoints"

generator_weights = None
discriminator_weights = None
if results_folder.is_dir():
    generator_weights = results_folder / "checkpoints/G_checkpoint.h5"
    discriminator_weights = results_folder / "checkpoints/D_checkpoint.h5"
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

dataset = getDataset(batch_size=batch_size, folder=dataset_folder, resolution=resolution,
                     use_flip_augmentation=True, image_extension=args.img_extension, shuffle_buffer_size=500)

G = Generator(resolution)
sample_G_output = G.initialize(batch_size)
if generator_weights is not None:
    G.built = True
    G.load_weights(generator_weights)
    print("Weights are loaded for G")
print(f"[Model G] output shape: {sample_G_output.shape}")

D = Discriminator(resolution)
sample_D_output = D.initialize(batch_size)
if discriminator_weights is not None:
    D.built = True
    D.load_weights(discriminator_weights)
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
test_input_for_generation = getInputNoise(test_input_size)
test_images = getTestImages(test_input_size, dataset_folder, resolution, args.img_extension)

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

signal.signal(signal.SIGINT, signalHandler)

fid_score_best = 5000

for epoch in range(start_epoch, epochs):
    start = time.perf_counter()
    print(f"Epoch {epoch} -------------")

    # TRAINING PHASE
    for step, image_batch in enumerate(dataset):
        G_loss, D_loss, D_real_fake_loss, D_I_reconstruction_loss, D_I_part_reconstruction_loss = train(
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

    # TESTING PHASE
    if args.fid:
        if epoch % args.fid_frequency == 0:
            fid_score = evaluate(inception_model=fid_inception_model,
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

    elapsed = time.perf_counter() - start
    print(f"Epoch number: {epoch} - "
          f"G loss value: {G_loss_metric.result():.4f} | "
          f"D loss value: {D_loss_metric.result():.4f} | "
          f"D real_fake loss value: {D_real_fake_loss_metric.result():.4f} | "
          f"D I recon loss: {D_I_reconstruction_loss_metric.result():.4f} | "
          f"D I part recon loss: {D_I_part_reconstruction_loss_metric.result():.4f} | "
          f"Elapsed time:{elapsed:.3f} seconds")

    G_loss_metric.reset_states()
    D_loss_metric.reset_states()
    D_real_fake_loss_metric.reset_states()
    D_I_part_reconstruction_loss_metric.reset_states()
    D_I_reconstruction_loss_metric.reset_states()

    # SAVE THE FID SCORE ONLY IF IS BETTER (LOWER) THAN THE PREVIOUS SAVED ONE
    if args.fid:
        if fid_score < fid_score_best:
            fid_score_best = fid_score
            G.save_weights(str(checkpoints_folder / "G_checkpoint.h5"))
            D.save_weights(str(checkpoints_folder / "D_checkpoint.h5"))

    # Generate test images
    generated_images = G(test_input_for_generation, training=False)
    generated_images = denormalizeImages(generated_images, dtype=tf.uint8).numpy()
    saveImages(epoch, generated_images, results_folder / "generated_images", 5, 5)

    # Generate reconstructions from Discriminator
    _, decoded_images, decoded_part_images = D(test_images, training=False)
    decoded_images = denormalizeImages(decoded_images, dtype=tf.uint8).numpy()
    decoded_part_images = denormalizeImages(decoded_part_images, dtype=tf.uint8).numpy()
    saveImages(epoch, decoded_images, results_folder / "reconstructed_whole_images", 5, 5)
    saveImages(epoch, decoded_part_images,
               results_folder / "reconstructed_part_images", 5, 5)

    if epoch == epochs - 1:
        file = open(checkpoints_folder / "epoch_file.txt", "w+")
        file.write(str(epoch))
        file.close()
