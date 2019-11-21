import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

"""
TODO:
Test critic vs cnn at each epoch
setup the data collection/ retraining
sort by / attach confidence to each generated img with epoch
"""


class generated_img(object):
    database = {'IMG':[],
                'Conf':[],
                'Epoch':[]}

    def __init__(self, img, epoch, confidence):
        self.epoch = epoch
        self.img = img
        self.confidence = confidence

    def show(self):
        ...

    def save_img(self):
        ...

    def add_to_database(self):
        print(self.img)
        print(self.confidence)
        generated_img.database['IMG'].append(self.img.numpy())
        generated_img.database['Conf'].append(self.confidence.numpy())
        generated_img.database['Epoch'].append(self.epoch)

    def save_db(self):
        df = pd.DataFrame(data = generated_img.database)
        df.to_csv('Data/generated_imgs/generated_imgs.csv')


class GAN(object):
    def __init__(self):
        self.artist = self.build_artist()
        self.critic = self.build_critic()
        self.artist_opt = tf.keras.optimizers.Adam(1e-4)
        self.critic_opt = tf.keras.optimizers.Adam(1e-4)

    def build_artist(self):
        model = Sequential([
            Dense((20 * 20 * 80), use_bias=False, input_shape=(100,)),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((20, 20, 80)),
            Conv2DTranspose(128, (4, 4), use_bias=False, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, (4, 4), strides=(2, 2), use_bias=False, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(1, (4, 4), strides=(2, 2), use_bias=False, padding='same')
        ])
        return model

    def build_critic(self):
        model = Sequential([
            Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[80, 80, 1]),
            LeakyReLU(),
            MaxPooling2D(),
            Dropout(.5),
            Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            LeakyReLU(),
            Dropout(.3),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1)
        ])
        return model

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def test_artist(self, seed=43):
        noise = tf.random.normal([1, 100])
        generated_image = self.artist(noise, training=False)
        arr = np.array(generated_image)
        print(arr.shape)
        plt.imshow(generated_image[0, :, :, 0], cmap='gray');
        return arr

    def test_critic(self, input):
        decision = self.critic(input)
        print(decision)

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
        # refactor
        generator = self.artist
        discriminator = self.critic
        BATCH_SIZE = 16
        noise_dim = 100

        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        self.artist_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        self.critic_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)
        gen_objects = []
        for conf, img in zip(self.critic(predictions, training=False), predictions):
            temp = generated_img(img, epoch, conf)
            temp.add_to_database()




        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('visualizations/image_at_epoch_{:04d}.png'.format(epoch))
        temp.save_db()

    def train(self, dataset, epochs):
        checkpoint_dir = 'checkpoints/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.artist_opt,
                                         discriminator_optimizer=self.critic_opt,
                                         generator=self.artist,
                                         discriminator=self.critic)
        noise_dim = 100
        num_examples_to_generate = 16
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            self.generate_and_save_images(self.artist, epoch + 1, seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        # display.clear_output(wait=True)
        self.generate_and_save_images(self.artist, epochs, seed)
