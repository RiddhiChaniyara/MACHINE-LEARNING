# importing the necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# storing the dataset path
clothing_fashion_mnist = tf.keras.datasets.fashion_mnist

# loading the dataset from tensorflow
(x_train, y_train),(x_test, y_test) = clothing_fashion_mnist.load_data()

# displaying the shapes of training and testing dataset
print('Shape of training cloth images: ',x_train.shape)

print('Shape of training label: ',y_train.shape)

print('Shape of test cloth images: ',x_test.shape)

print('Shape of test labels: ',y_test.shape)

# storing the class names as it is
# not provided in the dataset
label_class_names = ['T-shirt/top', 'Trouser',
					'Pullover', 'Dress', 'Coat',
					'Sandal', 'Shirt', 'Sneaker',
					'Bag', 'Ankle boot']

# display the first images
plt.imshow(x_train[0])
plt.colorbar() # to display the colourbar
plt.show()

plt.figure(figsize=(15, 5)) # figure size
i = 0
while i < 20:
	plt.subplot(2, 10, i+1)

	# showing each image with colourmap as binary
	plt.imshow(x_train[i], cmap=plt.cm.binary)

	# giving class labels
	plt.xlabel(label_class_names[y_train[i]])
	i = i+1

plt.show() # plotting the final output figure
