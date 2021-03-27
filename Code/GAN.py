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

        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(filters=filters * 2, kernel_size=(4, 4), strides=(1, 1))
        self.normalization = tf.keras.layers.BatchNormalization()
        self.glu = GLU()

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input tensors"""
        x = self.conv2d_transpose(inputs)
        x = self.normalization(x)
        x = self.glu(x)
        return x


class UpsamplingBlock(tf.keras.layers.Layer):
    def __init__(self, output_filters: int, **kwargs):
        super().__init__(**kwargs)
        self.output_filters = output_filters

        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv2d = tf.keras.layers.Conv2D(filters=output_filters * 2, kernel_size=(3, 3), padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.glu = GLU()

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input tensors"""
        x = self.upsampling(inputs)
        x = self.conv2d(x)
        x = self.normalization(x)
        x = self.glu(x)
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

    def __init__(self, input_low_res_filters: int, input_high_res_filters: int, **kwargs):
        super().__init__(**kwargs)

        self.pooling = tfa.layers.AdaptiveAveragePooling2D(output_size=(4, 4), data_format="channels_last")
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=input_low_res_filters, kernel_size=(4, 4), strides=1,
                                               padding="valid")
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=input_high_res_filters, kernel_size=(1, 1), strides=1,
                                               padding="valid")

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input tensors"""
        x_low, x_high = inputs

        x = self.pooling(x_low)
        x = self.conv2d_1(x)
        x = self.leaky_relu(x)
        x = self.conv2d_2(x)
        x = tf.nn.sigmoid(x)

        return x * x_high


class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same")

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input tensors"""
        x = self.conv(inputs)
        x = tf.nn.tanh(x)
        return x


class Generator(tf.keras.models.Model):
    """
    Input of the Generator is in shape: (B, 1, 1, 256)
    """

    def __init__(self, output_resolution: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert output_resolution in [256, 512, 1024], "Resolution should be 256 or 512 or 1024"
        if output_resolution not in [256, 512, 1024]:
            raise Exception("You have inserted a resolution value of %s. "
                            "Resolution should be 256 or 512 or 1024." % output_resolution)
        self.output_resolution = output_resolution

        self.input_block = GeneratorInputBlock(filters=1024)

        # Every layer is initiated, but we might not use the last ones. It depends on the resolution
        self.upsample_8 = UpsamplingBlock(512)
        self.upsample_16 = UpsamplingBlock(256)
        self.upsample_32 = UpsamplingBlock(128)
        self.upsample_64 = UpsamplingBlock(128)
        self.upsample_128 = UpsamplingBlock(64)
        self.upsample_256 = UpsamplingBlock(32)
        self.upsample_512 = UpsamplingBlock(16)
        self.upsample_1024 = UpsamplingBlock(8)

        self.sle_8_128 = SkipLayerExcitationBlock(self.upsample_8.output_filters, self.upsample_128.output_filters)
        self.sle_16_256 = SkipLayerExcitationBlock(self.upsample_16.output_filters, self.upsample_256.output_filters)
        self.sle_32_512 = SkipLayerExcitationBlock(self.upsample_32.output_filters, self.upsample_512.output_filters)

        self.output_image = OutputBlock()

    def initialize(self, batch_size: int = 1):
        sample_input = tf.random.normal(shape=(batch_size, 1, 1, 256), mean=0, stddev=1.0, dtype=tf.float32)
        sample_output = self.call(sample_input)
        return sample_output

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """ performs the logic of applying the layer to the input tensors"""
        x = self.input_block(inputs)  # --> (B, 4, 4, 1024)

        x_8 = self.upsample_8(x)  # --> (B, 8, 8, 512)
        x_16 = self.upsample_16(x_8)  # --> (B, 16, 16, 256)
        x_32 = self.upsample_32(x_16)  # --> (B, 32, 32, 128)
        x_64 = self.upsample_64(x_32)  # --> (B, 64, 64, 128)

        x_128 = self.upsample_128(x_64)  # --> (B, 128, 128, 64)
        x_sle_128 = self.sle_8_128([x_8, x_128])  # --> (B, 128, 128, 64)

        x_256 = self.upsample_256(x_sle_128)  # --> (B, 256, 256, 32)
        x = self.sle_16_256([x_16, x_256])  # --> (B, 256, 256, 32)

        if self.output_resolution > 256:
            x_512 = self.upsample_512(x)  # --> (B, 512, 512, 16)
            x = self.sle_32_512([x_32, x_512])  # --> (B, 512, 512, 16)

            if self.output_resolution > 512:
                x = self.upsample_1024(x)  # --> (B, 1024, 1024, 8)

        image = self.output_image(x)  # --> (B, resolution, resolution, 3)
        return image


# DISCRIMINATOR
class DiscriminatorInputBlock(tf.keras.layers.Layer):
    def __init__(self, downsampling_factor: int, filters, **kwargs):
        super().__init__(**kwargs)
        if downsampling_factor not in [1, 2, 4]:
            raise Exception("downsampling_factor should be in [1,2,4]")

        conv_1_strides = 2
        conv_2_strides = 2

        if downsampling_factor <= 2:
            conv_2_strides = 1

        if downsampling_factor == 1:
            conv_1_strides = 1

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=conv_1_strides, padding="same")
        self.activation_1 = tf.keras.layers.LeakyReLU(0.1)
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=conv_2_strides, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.normalization(x)
        x = self.activation_2(x)
        return x


class DownsamplingBlock1(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding="same")
        self.normalization_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.LeakyReLU(0.1)

        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")
        self.normalization_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.normalization_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.activation_2(x)
        return x


class DownsamplingBlock2(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding="valid")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.pooling(inputs)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class DownsamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        self.down_1 = DownsamplingBlock1(filters)
        self.down_2 = DownsamplingBlock2(filters)

    def call(self, inputs, **kwargs):
        x_1 = self.down_1(inputs)
        x_2 = self.down_2(inputs)
        return x_1 + x_2


class SimpleDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, output_filters, **kwargs):
        super().__init__(**kwargs)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv = tf.keras.layers.Conv2D(filters=output_filters * 2, kernel_size=3, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.glu = GLU()

    def call(self, inputs, **kwargs):
        x = self.upsampling(inputs)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.glu(x)
        return x


class SimpleDecoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder_block_filter_sizes = [256, 128, 128, 64]
        self.decoder_blocks = [SimpleDecoderBlock(output_filters=x) for x in self.decoder_block_filter_sizes]
        self.conv_output = tf.keras.layers.Conv2D(3, 1, 1, padding="same")

    def call(self, inputs, **kwargs):
        x = inputs
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        x = self.conv_output(x)
        x = tf.nn.tanh(x)
        return x


class RealFakeOutputBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1)
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.1)
        self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=4)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.conv_2(x)
        return x


class Discriminator(tf.keras.models.Model):
    def __init__(self, input_resolution: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert input_resolution in [256, 512, 1024], "Resolution should be 256 or 512 or 1024"
        self.input_resolution = input_resolution

        downsampling_factor_dict = {256: 1, 512: 2, 1024: 4}
        input_block_filters_dict = {256: 8, 512: 16, 1024: 32}
        self.input_block = DiscriminatorInputBlock(filters=input_block_filters_dict[input_resolution],
                                                   downsampling_factor=downsampling_factor_dict[input_resolution])

        self.downsample_128 = DownsamplingBlock(filters=64)
        self.downsample_64 = DownsamplingBlock(filters=128)
        self.downsample_32 = DownsamplingBlock(filters=128)
        self.downsample_16 = DownsamplingBlock(filters=256)
        self.downsample_8 = DownsamplingBlock(filters=512)

        self.decoder_image_part = SimpleDecoder()
        self.decoder_image = SimpleDecoder()

        self.real_fake_output = RealFakeOutputBlock(filters=256)

    def initialize(self, batch_size: int = 1):
        sample_input = tf.random.uniform(shape=(batch_size, self.input_resolution, self.input_resolution, 3), minval=0,
                                         maxval=1, dtype=tf.float32)
        sample_output = self.call(sample_input)
        return sample_output

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.input_block(inputs)  # --> (B, 256, 256, F)

        x = self.downsample_128(x)  # --> (B, 128, 128, 64)
        x = self.downsample_64(x)  # --> (B, 64, 64, 128)
        x = self.downsample_32(x)  # --> (B, 32, 32, 128)
        x_16 = self.downsample_16(x)  # --> (B, 16, 16, 256)
        x_8 = self.downsample_8(x_16)  # --> (B, 8, 8, 512)

        center_cropped_x_16 = cropCenterImages(x_16, 8)  # --> (B, 8, 8, 64)
        x_image_decoded_128_center_part = self.decoder_image_part(center_cropped_x_16)  # --> (B, 128, 128, 3)
        x_image_decoded_128 = self.decoder_image(x_8)  # --> (B, 128, 128, 3)

        x_real_fake_logits = self.real_fake_output(x_8)  # --> (B, 5, 5, 1)

        return x_real_fake_logits, x_image_decoded_128, x_image_decoded_128_center_part
