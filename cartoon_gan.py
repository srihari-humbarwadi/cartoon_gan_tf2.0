import tensorflow as tf
from glob import glob
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('TensorFLow', tf.__version__)


H, W = 256, 256
filters = 64
output_stride = 16
h_output = H // output_stride
w_output = W // output_stride
batch_size = 64
latent_dim = 100
display_noise = tf.random.normal(shape=[16, latent_dim], mean=0, stddev=1)
w_init = tf.initializers.glorot_uniform()


def load_data(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img)
    img = tf.image.resize(img, size=[H, W])[..., :3]
    img /= 127.5
    img -= 1
    return img


image_list = glob('cartoonset100k/*/*.png')
print('Found {} images'.format(len(image_list)))

image_dataset = tf.data.Dataset.from_tensor_slices(image_list)
image_dataset = image_dataset.shuffle(buffer_size=10240)
image_dataset = image_dataset.map(
    load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
image_dataset = image_dataset.batch(batch_size)
image_dataset = image_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)


def deconv_block(input_tensor, num_filters, kernel_size, strides, bn=True):
    x = tf.keras.layers.Conv2DTranspose(filters=num_filters,
                                        kernel_initializer=w_init,
                                        kernel_size=kernel_size,
                                        padding='same',
                                        strides=strides, use_bias=False if bn else True)(input_tensor)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x


def conv_block(input_tensor, num_filters, kernel_size, padding='same', strides=2, bn=True, activation=True):
    x = tf.keras.layers.Conv2D(filters=num_filters,
                               kernel_initializer=w_init,
                               kernel_size=kernel_size,
                               padding=padding,
                               strides=strides, use_bias=False if bn else True)(input_tensor)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


def build_generator(latent_dim=100):
    f = [2**i for i in range(5)][::-1]
    noise = tf.keras.layers.Input(
        shape=(latent_dim,), name='generator_noise_input')
    x = tf.keras.layers.Dense(f[0] * filters * h_output * w_output)(noise)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape(
        target_shape=[h_output, w_output, 16 * filters])(x)
    for i in range(1, 5):
        x = deconv_block(x,
                         num_filters=f[i] * filters,
                         kernel_size=5,
                         strides=2,
                         bn=True)
    x = deconv_block(x,
                     num_filters=3,
                     kernel_size=3,
                     strides=1,
                     bn=False)
    fake_output = tf.keras.layers.Activation(
        'tanh', name='generator_output')(x)
    return tf.keras.Model(inputs=[noise],
                          outputs=[fake_output],
                          name='Generator')


def build_discriminator():
    image_input = tf.keras.layers.Input(
        shape=[H, W, 3], name='discriminator_image_input')
    f = [2**i for i in range(4)]
    x = conv_block(
        image_input, num_filters=f[0] * filters, kernel_size=5, strides=2, bn=False)
    for i in range(1, 4):
        x = conv_block(x,
                       num_filters=f[i] * filters,
                       kernel_size=5,
                       strides=2,
                       bn=True)
    x = conv_block(x,
                   num_filters=1,
                   kernel_size=h_output,
                   padding='valid',
                   strides=1,
                   bn=False, activation=False)
    classification_logits = tf.keras.layers.Reshape(target_shape=[1])(x)
    return tf.keras.Model(inputs=[image_input], outputs=[classification_logits], name='Discriminator')


generator = build_generator(latent_dim)
discriminator = build_discriminator()
bce_loss_fn = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, label_smoothing=0.1)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, beta_1=0.5)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)


def loss_D(real_logits, fake_logits):
    '''Discriminator loss'''
    real_loss = 0.5 * bce_loss_fn(tf.ones_like(real_logits), real_logits)
    fake_loss = 0.5 * bce_loss_fn(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def loss_G(fake_logits):
    '''Generator loss'''
    loss = bce_loss_fn(tf.ones_like(fake_logits), fake_logits)
    return loss


def loss_D_real(real_logits):
    '''Discriminator loss, real images'''
    real_loss = bce_loss_fn(tf.ones_like(real_logits), real_logits)
    return real_loss


def loss_D_fake(fake_logits):
    '''Discriminator loss, fake images'''
    fake_loss = bce_loss_fn(tf.zeros_like(fake_logits), fake_logits)
    return fake_loss


@tf.function
def training_step(images):
    noise = tf.random.normal(shape=[batch_size, latent_dim], mean=0, stddev=1)

    with tf.GradientTape() as r_tape:
        real_logits = discriminator(images, training=True)
        discriminator_loss_real = loss_D_real(real_logits)
        discriminator_gradients_real = r_tape.gradient(
            discriminator_loss_real, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
            zip(discriminator_gradients_real, discriminator.trainable_variables))

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_logits = discriminator(
            generator(noise, training=True), training=True)

        discriminator_loss_fake = loss_D_fake(fake_logits)
        generator_loss = loss_G(fake_logits)

        discriminator_gradients_fake = d_tape.gradient(
            discriminator_loss_fake, discriminator.trainable_variables)
        generator_gradients = g_tape.gradient(
            generator_loss, generator.trainable_variables)

        d_optimizer.apply_gradients(
            zip(discriminator_gradients_fake, discriminator.trainable_variables))
        g_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables))

    return generator_loss, discriminator_loss_real, discriminator_loss_fake


def save_generated_images(noise, epoch=None):
    images = generator(noise)
    images = tf.clip_by_value((images + 1) * 127.5, 0, 255).numpy()
    for i in range(16):
        cv2.imwrite(f'train_viz/{i}_{epoch}.png',
                    cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))


def train(epochs=30, save_every=3, steps=None):
    batch_losses = {'g_loss': [], 'd_loss': []}
    epoch_losses = {'g_loss': [], 'd_loss': []}
    for ep in range(epochs):
        running_loss = {'g_loss': [], 'd_loss': []}
        for step, images in enumerate(image_dataset):
            batch_g_loss, batch_d_loss_real, batch_d_loss_fake = training_step(images)
            batch_d_loss = batch_d_loss_real + batch_d_loss_fake
            running_loss['g_loss'].append(batch_g_loss.numpy())
            running_loss['d_loss'].append(batch_d_loss.numpy())
            if (step + 1) % 25 == 0:
                print(
                    f'||epoch {ep+1}/{epochs} step {step+1}/{steps}|G_LOSS : {batch_g_loss:.3f}|D_LOSS : {batch_d_loss:.3f}||')
            if (step + 1) % 250 == 0:
                save_generated_images(display_noise, epoch=f'{ep+1}_{step+1}')
            tf.summary.scalar("generator_batch_loss",
                              batch_g_loss, step=step + 1)
            tf.summary.scalar("discriminator_batch_loss_total",
                              batch_d_loss, step=step + 1)
            tf.summary.scalar("discriminator_batch_loss_real",
                              batch_d_loss_real, step=step + 1)
            tf.summary.scalar("discriminator_batch_loss_fake",
                              batch_d_loss_fake, step=step + 1)
            writer.flush()
        batch_losses['g_loss'].extend(running_loss['g_loss'])
        batch_losses['d_loss'].extend(running_loss['d_loss'])
        epoch_losses['g_loss'].append(np.mean(running_loss['g_loss']))
        epoch_losses['d_loss'].append(np.mean(running_loss['d_loss']))
        if (ep + 1) % save_every == 0:
            print(f'||saving weights for epoch : {ep+1}||')
            generator.save_weights(f'model_files/generator_weights_{ep+1}.h5')
            discriminator.save_weights(
                f'model_files/discriminator_weights_{ep+1}.h5')
    return batch_losses, epoch_losses


writer = tf.summary.create_file_writer('logs')
with writer.as_default():
    steps_per_epoch = len(image_list)//batch_size
    batch_losses, epoch_losses = train(25, 3, steps_per_epoch)
