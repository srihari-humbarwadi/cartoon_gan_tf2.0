import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm_notebook

np.random.seed(69)
print('TensorFlow', tf.__version__)


H, W = 256, 256
filters = 64
output_stride = 16
h_output = H // output_stride
w_output = W // output_stride
latent_dim = 100
w_init = tf.initializers.glorot_uniform()


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


def generate(noise):
    img = tf.clip_by_value((generator(noise) + 1) * 127.5, 0, 255).numpy()
    img = np.squeeze(np.uint8(img))
    return img


def interpolate(a, b, steps=50):
    '''linear interpolation from a->b'''
    vectors = np.zeros(shape=[steps, latent_dim])
    for i, alpha in enumerate(np.linspace(0, 1, num=steps)):
        vector = alpha * a + (1 - alpha) * b
        vectors[i] = vector[0]
    return tf.constant(vectors)


def generate_gif(a, b, filename='', steps=100):
    vectors = interpolate(a, b, steps)
    images = np.zeros(shape=[steps, H, W, 3], dtype=np.uint8)
    for i in tqdm_notebook(range(len(vectors))):
        images[i] = generate(vectors[i][None, ...])
    loop = np.concatenate([images, images[::-1, ...]])
    imageio.mimsave(f'outputs/{filename}.gif', loop)


generator = build_generator(latent_dim)
generator.load_weights('model_files/generator_weights_6.h5')

a = np.load('vectors/boy.npy')
b = np.load('vectors/boy_dark_glasses.npy')

generate_gif(a, b, filename='boy_boy', steps=75)
