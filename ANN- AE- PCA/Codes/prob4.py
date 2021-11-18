#P4_a
for name in dir():
    del globals()[name]
    
import pickle
from IPython.display import Image, SVG
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import time
import os

#import images from data folder
ad=os.getcwd()
ad=ad+'\data\\'
f = open('images.pckl', 'rb')
X=pickle.load(f)
f.close()

#preparing input images
X=X.astype('float32')/255.
X=X.reshape((len(X),np.prod(X.shape[1:])))

start_time=time.time()

# input dimension = 3072
input_dim = X.shape[1]
encoding_dim = 50

compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

autoencoder = Sequential()
autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.summary()


input_img = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[0]
encoder = Model(input_img, encoder_layer(input_img))

encoder.summary()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X, X,
                epochs=50000,
                batch_size=50,
                verbose=0,
                shuffle=True,
                validation_data=(X, X))

num_images = 4
np.random.seed(42)
random_test_images = np.random.randint(X.shape[0], size=num_images)

encoded_imgs = encoder.predict(X)
decoded_imgs = autoencoder.predict(X)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(X[image_idx].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(10, 5))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    
print("--- %s seconds ---" % (time.time() - start_time))


num_images = 4
# np.random.seed(42)
random_test_images = np.random.randint(X.shape[0], size=num_images)

encoded_imgs = encoder.predict(X)
decoded_imgs = autoencoder.predict(X)

fig=plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(X[image_idx].reshape(32, 32, 3))
#     plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(10, 5))
#     plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(32, 32, 3))
#     plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# fig.savefig('p4_b.svg',format='svg')

#_____________________
#P4_c_deep

MSE=np.sum((X-decoded_imgs)**2)/3072
print("Mean Square Error (MSE): %s" % MSE)

#_____________________
#P4_d_deep

X_noisy = X + np.random.normal(loc=0.0, scale=0.1, size=X.shape)
X_noisy = np.clip(X_noisy, 0., 1.)

num_images = 4
# np.random.seed(42)
random_test_images = np.random.randint(X.shape[0], size=num_images)

# Denoise test images
X_denoised = autoencoder.predict(X_noisy)

fig=plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(2, num_images, i + 1)
    plt.imshow(X_noisy[image_idx].reshape(32, 32, 3))
#     plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot reconstructed image
    ax = plt.subplot(2, num_images, num_images + i + 1)
    plt.imshow(X_denoised[image_idx].reshape(32, 32, 3))
#     plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# fig.savefig('p4_d.svg',format='svg')
