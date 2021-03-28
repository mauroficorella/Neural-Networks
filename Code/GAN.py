import tensorflow as tf
import tensorflow_addons as tfa
from dataset_utils import cropCenterImages


# GATED LINEAR UNIT
class GLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        channels = tf.shape(inputs)[-1]
        num_split_channels = channels // 2

        return inputs[:, :, :, :num_split_channels] * tf.nn.sigmoid(inputs[:, :, :, num_split_channels:])


# LOSSES
def discrReconstructionLoss(real_img, decoded_img):
    return tf.reduce_mean(tf.keras.losses.MAE(real_img, decoded_img))


def discrRealFakeLoss(real_fake_out_logits_real_imgs, real_fake_out_logits_fake_imgs):
    real_loss = tf.minimum(0.0, -1 + real_fake_out_logits_real_imgs)
    real_loss = -1 * tf.reduce_mean(real_loss)

    fake_loss = tf.minimum(0.0, -1 - real_fake_out_logits_fake_imgs)
    fake_loss = -1 * tf.reduce_mean(fake_loss)

    return real_loss + fake_loss


def generatorLoss(real_fake_out_logits_fake_imgs):
    return -1 * tf.reduce_mean(real_fake_out_logits_fake_imgs)


# GENERATOR
class GeneratorInputBlock(tf.keras.layers.Layer):
    """
    Input Block

    Input shape: (B, 1, 1, 256)
    Output shape: (B, 4, 4, 256)
    """

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv2D_t = tf.keras.layers.Conv2DTranspose(filters=filters * 2, kernel_size=(4, 4), strides=(1, 1))
        self.norm = tf.keras.layers.BatchNormalization()
        self.GLU = GLU()

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input tensors"""
        x = self.conv2D_t(inputs)
        x = self.norm(x)
        x = self.GLU(x)
        return x


class UpsamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filters_out: int, **kwargs):
        super().__init__(**kwargs)
        self.filters_out = filters_out

        self.upsampling2D = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv2D = tf.keras.layers.Conv2D(filters=filters_out * 2, kernel_size=(3, 3), padding="same")
        self.norm = tf.keras.layers.BatchNormalization()
        self.GLU = GLU()

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input tensors"""
        x = self.upsampling2D(inputs)
        x = self.conv2D(x)
        x = self.norm(x)
        x = self.GLU(x)
        return x


class SkipLayerExcitationBlock(tf.keras.layers.Layer):
    """
    Skip-Layer Excitation Block

    This block receives 2 feature maps, a high and a low resolution one. Then transforms the low resolution feature map
    and at the end it is multiplied along the channel dimension with the high resolution input.

    E.g.:
    Inputs:
        - High_res shape: (B, 128, 128, 64)
        - Low_res shape: (B, 8, 8, 512)
    Output:
        - shape: (B, 128, 128, 64)
    """

    def __init__(self, low_res_filters_in: int, high_res_filters_in: int, **kwargs):
        super().__init__(**kwargs)

        self.aa_pooling2D = tfa.layers.AdaptiveAveragePooling2D(output_size=(4, 4), data_format="channels_last")
        self.conv2D_1 = tf.keras.layers.Conv2D(filters=low_res_filters_in, kernel_size=(4, 4), strides=1,
                                               padding="valid")
        self.leaky_ReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv2D_2 = tf.keras.layers.Conv2D(filters=high_res_filters_in, kernel_size=(1, 1), strides=1,
                                               padding="valid")

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input tensors"""
        x_low, x_high = inputs

        x = self.aa_pooling2D(x_low)
        x = self.conv2D_1(x)
        x = self.leaky_ReLU(x)
        x = self.conv2D_2(x)
        x = tf.nn.sigmoid(x)

        return x * x_high


class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv2D = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same")

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input tensors"""
        x = self.conv2D(inputs)
        x = tf.nn.tanh(x)
        return x


class Generator(tf.keras.models.Model):
    """
    Input of the Generator is in shape: (B, 1, 1, 256)
    """

    def __init__(self, resolution_out: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert output_resolution in [256, 512, 1024], "Resolution should be 256 or 512 or 1024"

        # TODO controllare perche qui con resolution_out c'e questo controllo, dato che sta anche a riga 331
        if resolution_out not in [256, 512, 1024]:
            raise Exception("You have inserted a resolution value of %s. "
                            "Resolution should be 256 or 512 or 1024." % resolution_out)
        self.resolution_out = resolution_out

        self.G_input_block = GeneratorInputBlock(filters=1024)

        # Every layer is initiated, but we might not use the last ones. It depends on the resolution
        self.upsample_8 = UpsamplingBlock(512)
        self.upsample_16 = UpsamplingBlock(256)
        self.upsample_32 = UpsamplingBlock(128)
        self.upsample_64 = UpsamplingBlock(128)
        self.upsample_128 = UpsamplingBlock(64)
        self.upsample_256 = UpsamplingBlock(32)
        self.upsample_512 = UpsamplingBlock(16)
        self.upsample_1024 = UpsamplingBlock(8)

        self.sle_8_128 = SkipLayerExcitationBlock(self.upsample_8.filters_out, self.upsample_128.filters_out)
        self.sle_16_256 = SkipLayerExcitationBlock(self.upsample_16.filters_out, self.upsample_256.filters_out)
        self.sle_32_512 = SkipLayerExcitationBlock(self.upsample_32.filters_out, self.upsample_512.filters_out)

        self.img_out = OutputBlock()

    def initialize(self, batch: int = 1):
        in_sample = tf.random.normal(shape=(batch, 1, 1, 256), mean=0, stddev=1.0, dtype=tf.float32)
        out_sample = self.call(in_sample)
        return out_sample

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """ performs the logic of applying the layer to the input tensors"""
        x = self.G_input_block(inputs)  # --> (B, 4, 4, 1024)

        x_8 = self.upsample_8(x)  # --> (B, 8, 8, 512)
        x_16 = self.upsample_16(x_8)  # --> (B, 16, 16, 256)
        x_32 = self.upsample_32(x_16)  # --> (B, 32, 32, 128)
        x_64 = self.upsample_64(x_32)  # --> (B, 64, 64, 128)

        x_128 = self.upsample_128(x_64)  # --> (B, 128, 128, 64)
        x_sle_128 = self.sle_8_128([x_8, x_128])  # --> (B, 128, 128, 64)

        x_256 = self.upsample_256(x_sle_128)  # --> (B, 256, 256, 32)
        x = self.sle_16_256([x_16, x_256])  # --> (B, 256, 256, 32)

        if self.resolution_out > 256:
            x_512 = self.upsample_512(x)  # --> (B, 512, 512, 16)
            x = self.sle_32_512([x_32, x_512])  # --> (B, 512, 512, 16)

            if self.resolution_out > 512:
                x = self.upsample_1024(x)  # --> (B, 1024, 1024, 8)

        img = self.img_out(x)  # --> (B, resolution, resolution, 3)
        return img


# DISCRIMINATOR
class DiscriminatorInputBlock(tf.keras.layers.Layer):
    def __init__(self, downsampling_factor: int, filters, **kwargs):
        super().__init__(**kwargs)
        if downsampling_factor not in [1, 2, 4]:
            raise Exception("downsampling_factor should be in [1,2,4]")

        conv2D_1_st = 2
        conv2D_2_st = 2

        if downsampling_factor <= 2:
            conv2D_2_st = 1

        if downsampling_factor == 1:
            conv2D_1_st = 1

        self.conv2D_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=conv2D_1_st, padding="same")
        self.act_leaky_ReLU_1 = tf.keras.layers.LeakyReLU(0.1)
        self.conv2D_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=conv2D_2_st, padding="same")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.act_leaky_ReLU_2 = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv2D_1(inputs)
        x = self.act_leaky_ReLU_1(x)
        x = self.conv2D_2(x)
        x = self.batch_norm(x)
        x = self.act_leaky_ReLU_2(x)
        return x


class DownsamplingBlock1(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv2D_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding="same")
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.act_leaky_ReLU_1 = tf.keras.layers.LeakyReLU(0.1)

        self.conv2D_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.act_leaky_ReLU_2 = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv2D_1(inputs)
        x = self.batch_norm_1(x)
        x = self.act_leaky_ReLU_1(x)
        x = self.conv2D_2(x)
        x = self.batch_norm_2(x)
        x = self.act_leaky_ReLU_2(x)
        return x


class DownsamplingBlock2(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.a_pooling2D = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.conv2D = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding="valid")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.act_leaky_ReLU = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.a_pooling2D(inputs)
        x = self.conv2D(x)
        x = self.batch_norm(x)
        x = self.act_leaky_ReLU(x)
        return x


class DownsamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        self.down_block_1 = DownsamplingBlock1(filters)
        self.down_block_2 = DownsamplingBlock2(filters)

    def call(self, inputs, **kwargs):
        x_1 = self.down_block_1(inputs)
        x_2 = self.down_block_2(inputs)
        return x_1 + x_2


class SimpleDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters_out, **kwargs):
        super().__init__(**kwargs)
        self.upsampling2D = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv2D = tf.keras.layers.Conv2D(filters=filters_out * 2, kernel_size=3, padding="same")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.GLU = GLU()

    def call(self, inputs, **kwargs):
        x = self.upsampling2D(inputs)
        x = self.conv2D(x)
        x = self.batch_norm(x)
        x = self.GLU(x)
        return x


class SimpleDecoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_block_filter_sizes = [256, 128, 128, 64]
        self.decoder_blocks = [SimpleDecoderBlock(filters_out=x) for x in self.decoder_block_filter_sizes]
        """self.decoder_blocks = []
        for x in self.decoder_block_filter_sizes:
            self.decoder_blocks = self.decoder_blocks.append([SimpleDecoderBlock(filters_out=x)])"""
        self.out_conv2D = tf.keras.layers.Conv2D(3, 1, 1, padding="same")

    def call(self, inputs, **kwargs):
        x = inputs
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        x = self.out_conv2D(x)
        x = tf.nn.tanh(x)
        return x


class RealFakeOutputBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv2D_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.act_leaky_ReLU = tf.keras.layers.LeakyReLU(0.1)
        self.conv2D_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=4)

    def call(self, inputs, **kwargs):
        x = self.conv2D_1(inputs)
        x = self.batch_norm(x)
        x = self.act_leaky_ReLU(x)
        x = self.conv2D_2(x)
        return x


class Discriminator(tf.keras.models.Model):
    def __init__(self, resolution_in: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if resolution_in not in [256, 512, 1024]:
            raise Exception("You have inserted a resolution value of %s. "
                            "Resolution should be 256 or 512 or 1024." % resolution_in)
        self.resolution_in = resolution_in

        downsampling_factor_dict = {256: 1, 512: 2, 1024: 4}
        D_input_block_filters_dict = {256: 8, 512: 16, 1024: 32}
        self.D_input_block = DiscriminatorInputBlock(filters=D_input_block_filters_dict[resolution_in],
                                                     downsampling_factor=downsampling_factor_dict[resolution_in])

        self.downsample_128 = DownsamplingBlock(filters=64)
        self.downsample_64 = DownsamplingBlock(filters=128)
        self.downsample_32 = DownsamplingBlock(filters=128)
        self.downsample_16 = DownsamplingBlock(filters=256)
        self.downsample_8 = DownsamplingBlock(filters=512)

        self.decoder_img_part = SimpleDecoder()
        self.decoder_img = SimpleDecoder()

        self.real_fake_out = RealFakeOutputBlock(filters=256)

    def initialize(self, batch: int = 1):
        in_sample = tf.random.uniform(shape=(batch, self.resolution_in, self.resolution_in, 3), minval=0,
                                      maxval=1, dtype=tf.float32)
        out_sample = self.call(in_sample)
        return out_sample

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.D_input_block(inputs)  # --> (B, 256, 256, F)

        x = self.downsample_128(x)  # --> (B, 128, 128, 64)
        x = self.downsample_64(x)  # --> (B, 64, 64, 128)
        x = self.downsample_32(x)  # --> (B, 32, 32, 128)
        x_16 = self.downsample_16(x)  # --> (B, 16, 16, 256)
        x_8 = self.downsample_8(x_16)  # --> (B, 8, 8, 512)

        center_cropped_x_16 = cropCenterImages(x_16, 8)  # --> (B, 8, 8, 64)
        x_img_decoded_128_center_part = self.decoder_img_part(center_cropped_x_16)  # --> (B, 128, 128, 3)
        x_img_decoded_128 = self.decoder_img(x_8)  # --> (B, 128, 128, 3)

        x_real_fake_logits = self.real_fake_out(x_8)  # --> (B, 5, 5, 1)

        return x_real_fake_logits, x_img_decoded_128, x_img_decoded_128_center_part
