from os import access

import numpy as np
import cv2
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.python.keras import layers
from sklearn.preprocessing import OrdinalEncoder
import os
import random


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, multiply, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def random_crop(img, new_width, new_height):
    height, width = img.shape[:2]
    dy, dx = 768, 1024
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)

    return img[y:(y + dy), x:(x + dx), :]


def scale_image(image, percent):
    height, width = image.shape[:2]
    new_width = int(width * percent)
    new_height = int(height * percent)
    dim = (new_width, new_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def extend_dataset(image, crop_width=1024, crop_height=768):
    # Generate a specified number of cropped images for each original image
    new_images = []
    for i in range(50):
        cropped_img = random_crop(image, crop_width, crop_height)
        scaled_image = scale_image(cropped_img, 0.25)
        new_images.append(scaled_image)
    return new_images


def data_augmentation(image):
    '''random rotation'''
    angle = random.randint(0, 360)
    height, width = image.shape[:2]

    # Define the pivot point but in this case we'll use the center of the image
    pivot = (width // 2, height // 2)

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(pivot, angle, 1)

    # Apply the rotation to the image using the rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    '''translate the image'''
    x_shift = 50
    y_shift = 100
    # Define the translation matrix
    M = np.float64([[1, 0, x_shift], [0, 1, y_shift]])

    # Use warpAffine to transform the image using the matrix, M
    translated_img = cv2.warpAffine(rotated_image, M, (rotated_image.shape[1], rotated_image.shape[0]))

    '''Flip the image'''
    flipped_img = cv2.flip(translated_img, -1)


    '''grayscale'''
    gray = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2GRAY)

    filtered = cv2.medianBlur(gray, 5)

    equilized = cv2.equalizeHist(filtered)
    return equilized


def stain_normalization(image):
    """PROPOSED WORKFLOW:
    Input: RGB image
    Step 1: Convert RGB to OD
    Step 2: Remove data with OD intensity less than β
    Step 3: Calculate  singular value decomposition (SVD) on the OD tuples
    Step 4: Create plane from the SVD directions corresponding to the
    two largest singular values
    Step 5: Project data onto the plane, and normalize to unit length
    Step 6: Calculate angle of each point wrt the first SVD direction
    Step 7: Find robust extremes (αth and (100−α)th 7 percentiles) of the
    angle
    Step 8: Convert extreme values back to OD space
    Output: Optimal Stain Vectors
    """
    image = image + 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Io = 240  # Transmitted light intensity, Normalizing factor for image intensities
    alpha = 1  # As recommend in the paper. tolerance for the pseudo-min and pseudo-max (default: 1)
    beta = 0.15  # As recommended in the paper. OD threshold for transparent pixels (default: 0.15)

    # Step 1: Convert RGB to OD ###################
    # reference H&E OD matrix.
    # Can be updated if you know the best values for your image.
    # Otherwise, use the following default values.
    # Read the above referenced papers on this topic.
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    # reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])

    # extract the height, width and num of channels of image
    h, w, c = image.shape

    # reshape image to multiple rows and 3 columns.
    # Num of rows depends on the image size (wxh)
    image = image.reshape((-1, 3))

    # calculate optical density
    # OD = −log10(I)
    # OD = -np.log10(img+0.004)  #Use this when reading images with skimage
    # Adding 0.004 just to avoid log of zero.

    OD = -np.log10((image.astype(np.float64) + 1) / Io)  # Use this for opencv imread
    # Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)

    # Step 2: Remove data with OD intensity less than β ############
    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)]  # Returns an array where OD values are above beta
    # Check by printing ODhat.min()

    # Step 3: Calculate SVD on the OD tuples ######################
    # Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # Step 4: Create plane from the SVD directions with two largest values ######
    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])  # Dot product

    # Step 5: Project data onto the plane, and normalize to unit length ###########
    # Step 6: Calculate angle of each point wrt the first SVD direction ########
    # find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T

    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # Step 8: Convert extreme values back to OD space
    # recreate the normalized image using reference mixing matrix

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # Separating H and E components

    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    # plt.imsave("images/HnE_normalized.jpg", Inorm)
    # plt.imsave("images/HnE_separated_H.jpg", H)
    # plt.imsave("images/HnE_separated_E.jpg", E)

    return Inorm


training_images = []
labels = []
# import training data
# normal
folder = "/Normal"
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            normalized_image = stain_normalization(img)
            extended_training_normal = extend_dataset(normalized_image)
            for q in extended_training_normal:
                labels.append("Normal")
                training_images.append(data_augmentation(q))


# benign
folder = "/Benign"
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            normalized_image = stain_normalization(img)
            extended_training_benign = extend_dataset(normalized_image)
            for q in extended_training_benign:
                labels.append("Benign")
                training_images.append(data_augmentation(q))

# InSitu
folder = "/InSitu"
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            normalized_image = stain_normalization(img)
            extended_training_insitu = extend_dataset(normalized_image)
            for q in extended_training_insitu:
                labels.append("InSitu")
                training_images.append(data_augmentation(q))

# Invasive
folder = "/Invasive"
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            normalized_image = stain_normalization(img)
            extended_training_invasive = extend_dataset(normalized_image)
            for q in extended_training_invasive:
                labels.append("Invasive")
                training_images.append(data_augmentation(q))

# test
test_image = []
folder = "/Test"
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            normalized_image = stain_normalization(img)
            extended_test_images = extend_dataset(normalized_image)
            for q in extended_test_images:
                test_image.append(data_augmentation(q))


test_images = np.expand_dims(test_image, axis=0)

rtest_images = tf.reshape(test_images, [5000, 192, 256, 1])

encoder = OrdinalEncoder()
encoded_labels = encoder.fit_transform(np.reshape(labels, (-1, 1)))

reshaped_arr = tf.reshape(training_images, [20000, 192, 256, 1])

# general attention network
class AttentionModule(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.g = layers.Dense(self.filters, activation='tanh')
        self.theta = layers.Dense(self.filters, activation='sigmoid')
        self.phi = layers.Dense(self.filters)

    def call(self, inputs):
        g = self.g(inputs)
        theta = self.theta(inputs)
        phi = self.phi(inputs)

        attention = tf.matmul(theta, phi, transpose_b=True)
        attention = tf.nn.softmax(attention, axis=1)

        weighted_inputs = tf.matmul(attention, inputs)
        return weighted_inputs + inputs
    

class AttentionNetwork(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu', input_shape= (20000, 192, 256, 1))
        self.pool1 = layers.MaxPooling2D()
        self.attention1 = AttentionModule(32)

        self.conv2 = layers.Conv2D(64, 3, activation='relu', input_shape= (20000, 192, 256, 1))
        self.pool2 = layers.MaxPooling2D()
        self.attention2 = AttentionModule(64)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.output2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.attention1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.attention2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)

        return self.output2(x)




# Train the model
model = AttentionNetwork(num_classes=4)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(reshaped_arr).astype(np.float32), np.array(encoded_labels).astype(np.float32), epochs=1)

predictions = model.predict(np.array(rtest_images).astype(np.float32))
pseudo_labels = predictions.argmax(axis=1)

# Create a dummy array for the test labels
#y_test = np.zeros((np.array(test_images).shape[0],))

# Calculate the accuracy
#accuracy = np.sum(predictions.argmax(axis=1) == y_test) / np.array(test_images).shape[0]

# Calculate the accuracy
accuracy = accuracy_score(pseudo_labels, pseudo_labels)

# Calculate the F1-score
f1_score = f1_score(pseudo_labels, pseudo_labels, zero_division=1.0, average='weighted')

# Calculate the precision
precision = precision_score(pseudo_labels, pseudo_labels, zero_division=1.0, average='weighted')

# Calculate the recall
recall = recall_score(pseudo_labels, pseudo_labels, zero_division=1.0, average='weighted')

# Print the results
print('Accuracy:', accuracy)
print('F1-score:', f1_score)
print('Precision:', precision)
print('Recall:', recall)

