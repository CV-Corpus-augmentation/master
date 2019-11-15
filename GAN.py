import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization

class GAN(object):
    def __init__(self):
        self.artist = self.build_artist()
        self.critic = self.build_critic()

    def build_artist(self):
        model = Sequential([
                        Dense((20*20*80), use_bias=False, input_shape=(100,)),
                        BatchNormalization(),
                        LeakyReLU(),
                        Reshape((20,20,80)),
                        Conv2DTranspose(128, (4,4), use_bias=False, padding = 'same'),
                        BatchNormalization(),
                        LeakyReLU(),
                        Conv2DTranspose(64, (4,4), strides=(2,2), use_bias=False, padding='same'),
                        BatchNormalization(),
                        LeakyReLU(),
                        Conv2DTranspose(1, (4,4), strides=(2,2), use_bias=False, padding='same')
                        ])
        return model

    def build_critic(self):
        model = Sequential([
                            Conv2D(32, (5,5), strides=(2,2), padding='same', input_shape=[80,80,1]),
                            LeakyReLU(),
                            MaxPooling2D(),
                            Dropout(.5),
                            Conv2D(64, (5,5), strides=(2,2), padding='same'),
                            LeakyReLU(),
                            Dropout(.3),
                            Flatten(),
                            Dense(256, activation='relu'),
                            Dense(1)
                            ])
        return model

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def train_step(self, images):

        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))