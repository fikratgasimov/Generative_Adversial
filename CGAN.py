from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D,  Dropout, Flatten, Dense, Input, Reshape
from keras.layers import Activation, Conv2DTranspose, UpSampling2D, BatchNormalization, Embedding, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist, fashion_mnist
from keras.optimizers import Adam, RMSprop
from PIL import Image
from tqdm import tqdm
from glob import glob
import os
import math
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt


class CGAN:

	def __init__(self, rows=64, cols=64, channels=3):
		self.rows = rows
		self.cols = cols
		self.channels = channels
		# self.shape = (28, 28, 1)
		self.shape = (self.rows, self.cols, self.channels)
		self.latent_size = 100
		self.sample_rows = 2
		self.sample_cols = 5
		self.sample_path = 'images'
		self.num_classes = 10

		optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
		
		
		image_shape = self.shape
		seed_size = self.latent_size
		
		
		# Get the discriminator and generator Models
		# Build and compile discriminator
		print("Build Discriminator")
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		
		# Build and Compile Generator
		print("Build Generator")
		self.generator = self.build_generator()
		self.generator.compile(loss = 'binary_crossentropy', optimizer= optimizer)

		# Random input for Generator
		random_input = Input(shape=(seed_size,))
		# Corresponding label
		label = Input(shape=(1,))	
		# Pass noise/random_input and label as input to the generator
		# this is generated image encompassing two variables
		generated_image = self.generator([random_input, label])
		
		# Put discriminator.trainable to False. We do not want to train the discriminator at this point in time
		self.discriminator.trainable = False
		
		# Validity takes generated images as input and determines validity
		validity = self.discriminator([generated_image, label])

		# Combined model(Stacked Generator and Discriminator)
		# as Random input => generates images => determines validity
		self.combined_model = Model([random_input, label], validity)
		self.combined_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	def build_generator(self):

		# Assging latent size to seed_size
		seed_size = self.latent_size

		model = Sequential()
		model.add(Dense(7 * 7 * 256, input_dim=seed_size))
		model.add(BatchNormalization(momentum=0.9))
		model.add(Activation('relu'))

		model.add(Reshape((7, 7, 256)))
		model.add(Dropout(0.4))

		model.add(Conv2DTranspose(128, (5, 5), padding='same'))
		model.add(BatchNormalization(momentum=0.9))
		model.add(Activation('relu'))
		model.add(UpSampling2D())

		model.add(Conv2DTranspose(64, (3, 3), padding='same'))
		model.add(BatchNormalization(momentum=0.9))
		model.add(Activation('relu'))
		model.add(UpSampling2D())

		model.add(Conv2DTranspose(32, (3, 3), padding='same'))
		model.add(BatchNormalization(momentum=0.9))
		model.add(Activation('relu'))

		model.add(Conv2DTranspose(1, (3, 3), padding='same'))
		model.add(Activation('sigmoid'))
		model.summary()

		noise = Input(shape=(seed_size,))
		label = Input(shape=(1,), dtype='int32')

		# Latent Input vector Z
		label_embeddings = Flatten()(Embedding(self.num_classes, self.latent_size)(label))
		input = multiply([noise, label_embeddings])
		generated_image = model(input)

		# build model from the input and output
		return (Model([noise, label], generated_image))
		
	def build_discriminator(self):
		# add input_shape, because we used BatchNormalization at first layer
		input_shape = self.shape

		model = Sequential()

		model.add(Conv2D(64, (3, 3), strides=2, padding='same', input_shape=input_shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.4))
		
		model.add(Conv2D(128, (3, 3), strides=2, padding='same'))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.4))
		
		model.add(Conv2D(256, (3,3), strides=2, padding='same'))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.4))
		
		model.add(Conv2D(512, (3,3), padding='same'))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.4))
		
		model.add(Flatten())
		
		model.add(Dense(1,activation='sigmoid'))
		model.summary()
		

		input_image = Input(shape=input_shape)
		label = Input(shape=(1,))
		
		label_embeddings = Flatten()(Embedding(self.num_classes, np.prod(self.shape))(label))
		flat_image = Flatten()(input_image)
		
		
		model_input = multiply([flat_image, label_embeddings])
		model_input = Reshape((64,64,3))(model_input)
	
		
		validity = model(model_input)
		
		return Model([input_image, label], validity)

	def get_image(self, image_path, width, height, mode):
		image = Image.open(image_path)
		image = image.resize([width,height])
		print("img",image)
		return np.array(image.convert(mode))

	def get_batch(self, image_files, width, height, mode):
		print(image_files)
		data_batch = np.array([self.get_image(sample_file, width, height, mode) for sample_file
							  in image_files])
		return data_batch

	def add_noise(self,image):
		ch = 3
		row,col = 64,64
		print(row,col,ch)
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean, sigma,(row, col, ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		plt.imshow(noisy)
		plt.show()
		print(noisy.shape)
		image = cv2.resize(noisy,(64,64))
		return image

	def plot(self, d_loss_logs_r_a, d_loss_logs_f_a, g_loss_logs_a):
		# Generate the plot at the end of training
		# Convert the log lists to numpy arrays
		d_loss_logs_r_a = np.array(d_loss_logs_r_a)
		d_loss_logs_f_a = np.array(d_loss_logs_f_a)
		g_loss_logs_a = np.array(g_loss_logs_a)
		plt.plot(d_loss_logs_r_a[:, 0], d_loss_logs_r_a[:, 1], label="Discriminator Loss - Real")
		plt.plot(d_loss_logs_f_a[:, 0], d_loss_logs_f_a[:, 1], label="Discriminator Loss - Fake")
		plt.plot(g_loss_logs_a[:, 0], g_loss_logs_a[:, 1], label="Generator Loss")
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.title('Variation of losses over epochs')
		plt.grid(True)
		plt.show()
	def train(self, epochs=10000, batch_size=128, save_freq=200):

		data_dir = "data/img_align_celeba"

		filepaths = os.listdir(data_dir)

		seed_size = self.latent_size
		half_batch = int(batch_size / 2)

		# Create lists for logging the losses
		d_loss_logs_r = []
		d_loss_logs_f = []
		g_loss_logs = []
		n_iterations = math.floor(len(filepaths) / batch_size)
		print_function(n_iterations)

		for epoch in range(epochs):

			# " Train Discriminator " #

			# Select a random half batch of images
			for ite in range(n_iterations):

				X_train = self.get_batch(glob(os.path.join(data_dir, '*.jpg'))
											[ite * batch_size:(ite + 1) * batch_size], 64, 64, 'RGB')

				# Normalizing this way
				X_train = (X_train.astype(np.float32) - 127.5) / 127.5
				X_train = np.array([self.add_noise(image) for image in X_train])
				print(X_train.shape[0])
				idx = np.random.rand(0, X_train.shape[0], half_batch)
				imgs = X_train[idx]
				noise = np.random.normal(0, 1, size=[half_batch, seed_size])
				X_fake = self.generator.predict(noise)

				# Train Disciminator
				d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
				d_loss_fake = self.discriminator.train_on_batch(X_fake, np.zeros((half_batch, 1)))
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				# Train Generator
				noise = np.random.normal(0, 1, size=[batch_size, seed_size])

				# Generate want Discriminator to label the genrated samples
				# as valid ones
				# Valid labels for generated images,
				sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
				valid_y = np.array([1] * batch_size)
				# due to maximizing Discriminator Loss
				g_loss = self.combined_model.train_on_batch([noise, sampled_labels], valid_y)

				print("%d %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, ite, d_loss[0],
																			 100 * d_loss[1], g_loss))

				# Append the logs with the loss values in each training step
				d_loss_logs_r.append([epoch, d_loss[0]])
				d_loss_logs_f.append([epoch, d_loss[1]])
				g_loss_logs.append([epoch, g_loss])

				d_loss_logs_r_a = np.array(d_loss_logs_r)
				d_loss_logs_f_a = np.array(d_loss_logs_f)
				g_loss_logs_a = np.array(g_loss_logs)

				# If at save_frequency => save generated image samples
				if ite % save_freq == 0:
					self.save_imgs(epoch, ite)
					plt.plot(d_loss_logs_r_a[:, 0], d_loss_logs_r_a[:, 1], label="Discriminator Loss-Real")
					plt.plot(d_loss_logs_f_a[:, 0], d_loss_logs_f_a[:, 1], label="Discriminator Loss-Fake")
					plt.plot(g_loss_logs_a[:, 0], g_loss_logs_a[:, 1], label="Generator Loss")
					plt.xlabel('Epoch-iterations')
					plt.ylabel('Loss')
					plt.legend()
					plt.title('Variation of loss over epochs')
					plt.grid(True)
					plt.show()

			model_json = self.generator.to_json()
			with open("model" + str(epoch) + ".json", "w") as json_file:
				json_file.write(model_json)
				self.generator.save_weights("model" + str(epoch) + ".h5")
				print("Save model to disk")



		

	def save_imgs(self, epoch,noise):
		r, c = self.sample_rows, self.sample_cols
		
		sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)
	
		gen_imgs = self.generator.predict([noise, sampled_labels])
		
		filename = os.path.join(self.sample_path,'%d.png'% epoch)
		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i, j].imshow(gen_imgs[cnt, :,:,0])
				axs[i, j].axis('off')
				cnt += 1
		fig.savefig(filename)
		plt.close()

if __name__ == '__main__':
	cgan = CGAN()
	cgan.train(epochs=6,batch_size=32, save_freq=200)

