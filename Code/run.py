from pathlib import Path
from dataset_utils import *
from GAN import Generator, Discriminator
from FID_utils import InceptionModel
from training import train, evaluate
from utils import saveImages
import argparse
import sys
import signal
import time
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument("--img_extension", type=str, default="jpg",
                    help="Extension of the images used for the training (png, jpg, jpeg). Default is jpg.")
parser.add_argument("--resolution", type=int, default=256,
                    help="Resolution can be either 256, 512 or 1024. Default resolution value is 256.")
parser.add_argument("--batch", type=int, default=1,
                    help="Batch size to use (if you have a GPU with less than 8GB of VRAM, set this value"
                         "either on 1,2 or 4).")
parser.add_argument("--epochs", type=int, default=1000,
                    help="Epoch number to use during the training.")
parser.add_argument("--dataset-folder", type=str, required=True, help="Folder containing the dataset images.")
parser.add_argument("--out-folder-name", type=str, default="Experiment",
                    help="Name of the output folder regarding the current experiment.")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrites previous output folders with same name if already exists.")
parser.add_argument("--G-learning-rate", type=float, default=2e-4, help="Generator learning rate.")
parser.add_argument("--D-learning-rate", type=float, default=2e-4, help="Discriminator learning rate.")
parser.add_argument("--diff-augment", action="store_true",
                    help="Tell whether to apply or not differentiable augmentation.")
parser.add_argument("--FID", action="store_true", help="This tells whether to evaluate or not the FID value.")
parser.add_argument("--FID-freq", type=int, default=1,
                    help="This tells at what frequency (in terms of epochs) the FID value will be evaluated.")
parser.add_argument("--FID-num-of-imgs", type=int, default=128,
                    help="This tells how many images will be used for the FID value calculation.")

args = parser.parse_args()
print(args)

img_extension = args.img_extension
resolution = args.resolution
batch = args.batch
epochs = args.epochs
dataset_folder = args.dataset_folder
G_learning_rate = args.G_learning_rate
D_learning_rate = args.D_learning_rate
FID = args.FID
FID_freq = args.FID_freq
FID_num_of_imgs = args.FID_num_of_imgs
diff_augment = args.diff_augment


def signalHandler(sig, frame):
    print('You pressed Ctrl+C!')
    epoch_file = open(checkpoints_folder / "epoch_file.txt", "w")
    epoch_file.write(str(epoch))
    epoch_file.close()
    sys.exit(0)


GPUs = tf.config.list_physical_devices("GPU")
_ = [tf.config.experimental.set_memory_growth(x, True) for x in GPUs]


G_weights = None
D_weights = None

out_folder = Path("Results") / args.out_folder_name
checkpoints_folder = out_folder / "Checkpoints"
if out_folder.is_dir():
    G_weights = out_folder / "Checkpoints/G_checkpoint.h5"
    D_weights = out_folder / "Checkpoints/D_checkpoint.h5"
    with open(checkpoints_folder / "epoch_file.txt") as f:
        start_epoch = int(f.read())
if not checkpoints_folder.is_dir():
    checkpoints_folder.mkdir(parents=True)

tensorboard_logs_folder = out_folder / "Tensorboard_logs"
if not tensorboard_logs_folder.is_dir():
    tensorboard_logs_folder.mkdir(parents=True)


dataset = getDataset(batch=batch, dataset_folder=dataset_folder, resolution=resolution,
                     flip_augment=True, img_extension=args.img_extension, size_buff_shuffle=500)

G = Generator(resolution)
G_out = G.initialize(batch)
if G_weights is not None:
    G.built = True
    G.load_weights(G_weights)
    print("G weights loaded successfully.")
else:
    start_epoch = 0
print(f"[G] Output shape: {G_out.shape}.")

D = Discriminator(resolution)
D_out = D.initialize(batch)
if D_weights is not None:
    D.built = True
    D.load_weights(D_weights)
    print("D weights loaded successfully.")
else:
    start_epoch = 0
print(f"[D] Real_fake output shape: {D_out[0].shape}.")
print(f"[D] Image output shape: {D_out[1].shape}.")
print(f"[D] Image part output shape: {D_out[2].shape}.")

G_optim = tf.optimizers.Adam(learning_rate=G_learning_rate)
D_optim = tf.optimizers.Adam(learning_rate=D_learning_rate)

if FID:
    # Model for the FID calculation
    FID_inception_model = InceptionModel(height=resolution, width=resolution)

test_input_size = 25
test_input_generation = getInputNoise(test_input_size)
test_images = getTestImages(test_input_size, dataset_folder, resolution, img_extension)

tensorboard_file_writer = tf.summary.create_file_writer(str(tensorboard_logs_folder))
tensorboard_file_writer.set_as_default()

G_loss_metric = tf.keras.metrics.Mean()
D_loss_metric = tf.keras.metrics.Mean()
D_real_fake_loss_metric = tf.keras.metrics.Mean()
D_I_reconstruction_loss_metric = tf.keras.metrics.Mean()
D_I_part_reconstruction_loss_metric = tf.keras.metrics.Mean()

diff_augment_policies = None
if diff_augment:
    diff_augment_policies = "color,translation,cutout"

signal.signal(signal.SIGINT, signalHandler)

FID_score_best = 5000

for epoch in range(start_epoch, epochs):
    start = time.perf_counter()
    print(f"Epoch {epoch} -------------")

    # TRAINING PHASE
    for step, batch_img in enumerate(dataset):
        G_loss, D_loss, D_real_fake_loss, D_I_reconstruction_loss, D_I_part_reconstruction_loss = train(
            G=G,
            D=D,
            optim_G=G_optim,
            optim_D=D_optim,
            images=batch_img,
            diff_augm_policies=diff_augment_policies)

        G_loss_metric(G_loss)
        D_loss_metric(D_loss)
        D_real_fake_loss_metric(D_real_fake_loss)
        D_I_reconstruction_loss_metric(D_I_reconstruction_loss)
        D_I_part_reconstruction_loss_metric(D_I_part_reconstruction_loss)

    # TESTING PHASE
    if FID:
        if epoch % FID_freq == 0:
            FID_score = evaluate(inception_model=FID_inception_model,
                                 dataset=dataset,
                                 G=G,
                                 batch=batch,
                                 img_height=resolution,
                                 img_width=resolution,
                                 num_imgs=FID_num_of_imgs)
            print(f"[FID] {FID_score:.2f}.")
            tf.summary.scalar("FID_score", FID_score, epoch)

    tf.summary.scalar("G_loss/G_loss", G_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_loss", D_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_real_fake_loss", D_real_fake_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_I_reconstruction_loss", D_I_reconstruction_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_I_part_reconstruction_loss", D_I_part_reconstruction_loss_metric.result(), epoch)

    elapsed = time.perf_counter() - start
    print(f"Epoch number: {epoch} - "
          f"G Loss value: {G_loss_metric.result():.4f} | "
          f"D Loss value: {D_loss_metric.result():.4f} | "
          f"D Real_fake loss value: {D_real_fake_loss_metric.result():.4f} | "
          f"D I recon loss: {D_I_reconstruction_loss_metric.result():.4f} | "
          f"D I part recon loss: {D_I_part_reconstruction_loss_metric.result():.4f} | "
          f"Elapsed time: {elapsed:.3f} seconds.")

    G_loss_metric.reset_states()
    D_loss_metric.reset_states()
    D_real_fake_loss_metric.reset_states()
    D_I_part_reconstruction_loss_metric.reset_states()
    D_I_reconstruction_loss_metric.reset_states()

    # SAVE THE WEIGHTS ONLY IF THE FID SCORE IS BETTER (LOWER) THAN THE PREVIOUS ONE
    if FID:
        if FID_score < FID_score_best:
            FID_score_best = FID_score
            G_checkpoint_path = str(checkpoints_folder / "G_checkpoint.h5")
            D_checkpoint_path = str(checkpoints_folder / "D_checkpoint.h5")
            G.save_weights(G_checkpoint_path)
            D.save_weights(D_checkpoint_path)

    # Generate test images
    generated_imgs = G(test_input_generation, training=False)
    generated_imgs = denormalizeImages(generated_imgs, dtype=tf.uint8).numpy()
    saveImages(epoch, generated_imgs, out_folder / "Generated_images", 5, 5)

    # Generate reconstructions from Discriminator
    _, decoded_imgs, decoded_part_imgs = D(test_images, training=False)
    decoded_imgs = denormalizeImages(decoded_imgs, dtype=tf.uint8).numpy()
    decoded_part_imgs = denormalizeImages(decoded_part_imgs, dtype=tf.uint8).numpy()
    saveImages(epoch, decoded_imgs, out_folder / "Reconstructed_whole_images", 5, 5)
    saveImages(epoch, decoded_part_imgs,
               out_folder / "Reconstructed_part_images", 5, 5)

    if epoch == epochs - 1:
        file = open(checkpoints_folder / "epoch_file.txt", "w+")
        file.write(str(epoch))
        file.close()
