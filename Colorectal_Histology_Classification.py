#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pickle

#!pip install tensorflow-transform
#import tensorflow_transform as tft

# evaluate pca with logistic regression algorithm for classification
# from numpy import mean
# from numpy import std
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.utils.extmath import randomized_svd

if not os.path.isdir('MATH4480'):
    get_ipython().system('git clone https://github.com/m-a-huber/MATH4480')

from MATH4480.project_utils import project_2_utils, project_3_utils, project_4_utils


# In[ ]:


dataset, info = tfds.load("colorectal_histology", as_supervised=True, with_info=True)

# Extract informative features
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes

print(class_names)
print(n_classes)


# In[ ]:


test_set, valid_set, train_set = tfds.load("colorectal_histology", 
                                           split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
                                           as_supervised=True)

print("Train set size: ", len(train_set)) 
print("Test set size: ", len(test_set))   
print("Valid set size: ", len(valid_set))


# In[ ]:


print(info)
print(tfds.show_examples(test_set, info, rows=4, cols=4))


# In[ ]:


def normalize_img(image, label):
  return tf.cast(image, tf.float32)/255.0, label

AUTOTUNE=tf.data.experimental.AUTOTUNE
BATCH_SIZE=16
train_set=train_set.map(normalize_img, num_parallel_calls=AUTOTUNE)
train_set=train_set.cache()
train_set=train_set.shuffle(info.splits["train"].num_examples)
train_set=train_set.batch(BATCH_SIZE)
train_set=train_set.prefetch(AUTOTUNE)

test_set=test_set.map(normalize_img, num_parallel_calls=AUTOTUNE)
test_set=test_set.batch(16)
test_set=test_set.prefetch(AUTOTUNE)

valid_set=valid_set.map(normalize_img, num_parallel_calls=AUTOTUNE)
valid_set=valid_set.batch(16)
valid_set=valid_set.prefetch(AUTOTUNE)


# In[ ]:


print(test_set)
print(train_set)
print(valid_set)


# In[ ]:


x_train = []
y_train = []  # store true labels
x_val = []
y_val = []
x_test= []
y_test = []


# iterate over the dataset
for image_batch, label_batch in train_set:   # use dataset.unbatch() with repeat
   # append true labels
   x_train.append(image_batch)
   y_train.append(label_batch)

for image_batch, label_batch in valid_set:   # use dataset.unbatch() with repeat
   # append true labels
   x_val.append(image_batch)
   y_val.append(label_batch)

for image_batch, label_batch in test_set:   # use dataset.unbatch() with repeat
   # append true labels
   x_test.append(image_batch)
   y_test.append(label_batch)

# convert the true and predicted labels into tensors
x_train = tf.concat([item for item in x_train], axis = 0)
y_train = tf.concat([item for item in y_train], axis = 0)
print(x_train.shape)
print(y_train)

x_val = tf.concat([item for item in x_val], axis = 0)
y_val = tf.concat([item for item in y_val], axis = 0)
print(x_val.shape)
print(y_val)

x_test = tf.concat([item for item in x_test], axis = 0)
y_test = tf.concat([item for item in y_test], axis = 0)
print(x_test.shape)
print(y_test)


# In[ ]:


x_train=np.array(x_train)
print(type(x_train))
x_train=np.reshape(x_train, (3750, -1))
print(x_train.shape, type(x_train))

x_val=np.array(x_val)
print(type(x_val))
x_val=np.reshape(x_val, (750, -1))
print(x_val.shape, type(x_val))

x_test=np.array(x_test)
print(type(x_test))
x_test=np.reshape(x_test, (500, -1))
print(x_test.shape, type(x_test))


# In[ ]:


U, S, Vt = randomized_svd(x_train, n_components=300)


# In[ ]:


Vt=np.transpose(Vt)
print(Vt.shape)
print(S)


# We see that the first 2 principal components capture a disproportionately high amount of the variance in the data set. It is computationally too expensive to compute all 67500 principal components and their associated eigenvectors, so we only have the first 300. We have good reason to believe that the first 300 principal components and their associated eigenvectors suffices to capture a reasonably high proportion of the total variance in the 67500 total variables, since the later principal components have decreased by orders of magnitudes compared to the first few. 

# We now wish to project `x_train` onto the subspace of $\mathbb{R}^{67500}$ spanned by the first 300 elements $v_1, v_2,...,v_{300}$ of the basis given by $Vt$.
# 
# Since that basis is orthonormal, the projection of any vector $w$ onto that subspace is obtained by simply taking the dot product. That is, $w$ projects to $(w\cdot v_1)v_1 + (w\cdot v_2)v_2+...+(w\cdot v_{300})v_{300}$.
# 
# We take the projections of `x_train` and `x_val` onto the subspace $Span(v_1, v_2,...,v_{300})$ $\leq$ $\mathbb{R}^{67500}$

# In[ ]:


x_train_reduced_300D = np.matmul(x_train, Vt)
x_val_reduced_300D = np.matmul(x_val, Vt)
x_test_reduced_300D = np.matmul(x_test, Vt)


# In[ ]:


print(x_train_reduced_300D.shape)
print(x_val_reduced_300D.shape)
print(x_test_reduced_300D.shape)


# In[ ]:


plt.scatter(x_train_reduced_300D[:,0], x_train_reduced_300D[:,1], c=y_train)


# It seems projection onto the subspace of $\mathbb{R}^{67500}$ defined by the span of the first 2 principal component vectors already allows for some semblance of separability by some boundaries (that are not necessarily linear hyperplanes)

# In[ ]:


x_train_reduced_3D = np.matmul(x_train, Vt[:,:3])


# In[ ]:


x_train_reduced_3D.shape


# In[ ]:


import plotly.graph_objects as go

x = x_train_reduced_3D[:,0]
y = x_train_reduced_3D[:,1]
z = x_train_reduced_3D[:,2]

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers+text',
    marker=dict(
        size=8,
        color=y_train,                 
        opacity=0.8
    )
)])

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# Projection onto the span of the first 3 principal component vectors doesn't look much better though.

# We can now train a k-class (k=8) classification model on the 300D projection x_train_reduced_300D we obtained above.

# In[ ]:


def create_multi_classification_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(8, activation = 'softmax'))
    return model


# In[ ]:


tf.keras.backend.clear_session()

########## Your Code goes here #############

# TODO: Create the model
model = create_multi_classification_model((300,))

# TODO: Create the optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.005)

# TODO: Create the loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# TODO: Create a binary accuracy metric
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

# TODO: Compile the model
model.compile(optimizer=opt, loss=loss, metrics = metrics)

#############################################
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


# In[ ]:


history = model.fit(x_train_reduced_300D, y_train, batch_size = 32, epochs=100, validation_data= (x_val_reduced_300D, y_val))


# In[ ]:


tf.keras.utils.plot_model(model, show_shapes = True)


# In[ ]:


pd.DataFrame(history.history).plot()


# Our Dataset takes a long time to train; this is not surprising, since the input dimension is 150x150x3. Furthermore, our task is k-class (with k=8) classification on images (taken by medical imaging equipment) of tumors. By visual inspection, we see that the images tend to contain lots of whitespace, which is useless and can needlessly consume more memory and processing power, resulting in slower training. So, we used PCA to attempt to dimension-reduce the data. By our a priori heuristic-based analysis, we had good reason to expect significant reductions of input dimension; most of the useful information was concentrated only in a fraction of pixels near the center of the 150 x 150 x 3 cube.
# 
# When we ran the training with the above training parameters and hyperparameters, the best holistic result (balancing validation and training accuracy) was:
# loss: 0.5480 - sparse_categorical_accuracy: 0.7579 - val_loss: 1.0185 - val_sparse_categorical_accuracy: 0.6613

# Now we train our model on the input data without using PCA; this will take a long time, with relatively minor improvements in training and validatinon accuracy of the best model. 

# In[ ]:


def create_cnn(input_shape):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(layers.Flatten())
    #remove last 2 layers --> get flattened shape --> apply PCA to flattened shape --> create standard dense neural network to process pca-reduced design matrix (?)
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='softmax'))

    return model


# We can also try to utilize an autoencoder architecture in the dense neural network layers at the end of the convolutional architecture. 

# In[ ]:


def create_cnn_autoencoded(input_shape):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(layers.Flatten())
    
    #apply autoencoder to hopefully compress the necessary information in the dataset to a lower dimension
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    
    #output layer classifies image matrices into one of 8 categories
    model.add(tf.keras.layers.Dense(8, activation='softmax'))

    return model


# In[ ]:


tf.keras.backend.clear_session()

model = create_cnn((150,150,3))

opt = tf.keras.optimizers.SGD(learning_rate = 0.01)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

acc = tf.keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer=opt,
              loss=loss,
              metrics=[acc])


# In[ ]:


history = model.fit(train_set, epochs=20, batch_size = 16, validation_data=valid_set)


# With the above parameters and hyperparameters, the best model we obtained was:
# loss: 0.6945 - sparse_categorical_accuracy: 0.7259 - val_loss: 0.6256 - val_sparse_categorical_accuracy: 0.7893
# 
# We see that the validation accuracy is actually significantly better than what we obtained for the PCA-reduced best model from before. This could be due to the following factors:
# 
# 1). CNN is an inherently better architecture compared to the standard dense neural networks used with the PCA example from before
# 
# 2). The projection onto the first 300 principal component vectors does not capture enough of the dataset's variance
# 
# 3). The only problems are due to poorly-chosen #layers, #neurons in the layers, and hyperparameters; essentially, not enough trial and error

# In[ ]:


tf.keras.utils.plot_model(model, show_shapes = True)


# In[ ]:


model.summary()


# In[ ]:


tf.keras.backend.clear_session()

model = create_cnn_autoencoded((150,150,3))

opt = tf.keras.optimizers.SGD(learning_rate = 0.01)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

acc = tf.keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer=opt,
              loss=loss,
              metrics=[acc])


# In[ ]:


history = model.fit(train_set, epochs=20, batch_size = 16, validation_data=valid_set)


# With the autoencoder and its associated weights for each layer, it seems that the CNN does not perform very well. This might be able to be fixed after trial and error, but due to the exhorbitant amount of time required to train the model we did not have time to properly attempt the testing for many times.
# 
# In principle, the addition of the autoencoder should allow us to reduce the dimension of the features after the CNN layers have first been applied to the input data and the resulting reduced matrix has been flattened.

# In[ ]:


tf.keras.utils.plot_model(model, show_shapes = True)


# We now try something new. Instead ot running PCA on the input feature vectors and then applying only a standard dense neural network, we will modify our architecture to first run a Conv2D neural network on the unmodified input feature vectors with a a few filters of moderately-sized dimension, and then flatten the resulting reduced-size image data matrix. Our heuristic is that this should pick out some of the most relevant features in the images (we will use Max pooling). Given the flattened output from the Conv2D layers, we will then apply PCA to the output (now a design matrix where the rows are the flattened version of the now-reduced 2D image data whose cell entries are the most prominent local features in the original input data). Once this is done, we can apply the conventional dense-layered neural network as we would always do in a convolutional architecture.

# Unfortunately, I could not get this idea to work because I am unsure how to extract the flattened and reduced matrix after the convolutional layers step in the middle of the model; this would presumably be necessary to apply PCA.
# 
# The unsuccessful code is attached below.

# In[ ]:


def create_cnn_pca_preprocessing(input_shape):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(layers.Flatten())
    #remove last 2 layers --> get flattened shape --> apply PCA to flattened shape --> create standard dense neural network to process pca-reduced design matrix (?); 
    #does not work because validation/test dimensions don't match up with filter-reduced image matrix from convolutional layers
    
    #alt method (?): try to get the matrix directly at this step --> apply PCA --> apply dense neural network layers as usual; 
    #however, this does not work either due to the reason stated in the comment block above
    return model


# In[ ]:


tf.keras.backend.clear_session()

model = create_cnn_pca_preprocessing((150,150,3))

opt = tf.keras.optimizers.SGD(learning_rate = 0.01)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

acc = tf.keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer=opt,
              loss=loss,
              metrics=[acc])


# In[ ]:


history = model.fit(train_set, epochs=5, batch_size = 16, validation_data=valid_set)


# In[ ]:


prominent_features_flattened=model.predict(train_set)


# In[ ]:


print(prominent_features_flattened.shape,type(prominent_features_flattened))


# In[ ]:


U, S, Vt = randomized_svd(prominent_features_flattened, n_components=32)


# In[ ]:


Vt=np.transpose(Vt)
print(Vt.shape)
print(S)


# In[ ]:


prominent_features_flattened_reduced_10D = np.matmul(prominent_features_flattened, Vt[:,:10])


# In[ ]:


print(prominent_features_flattened_reduced_10D.shape)


# In[ ]:


def create_multi_classification_model_2(input_shape):
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(8, activation = 'softmax'))
    return model


# In[ ]:


x_val_reduced_10D = np.matmul(x_val, Vt[:,:10])
print(prominent_features_flattened_reduced_10D.shape)
print(x_val_reduced_10D.shape)
print(y_train.shape)


# In[ ]:


tf.keras.backend.clear_session()

########## Your Code goes here #############

# TODO: Create the model
model = create_multi_classification_model_2((10,))

# TODO: Create the optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.005)

# TODO: Create the loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# TODO: Create a binary accuracy metric
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

# TODO: Compile the model
model.compile(optimizer=opt, loss=loss, metrics = metrics)

#############################################


# 

# In[ ]:


history = model.fit(prominent_features_flattened_reduced_10D, y_train, batch_size = 32, epochs=100, validation_data= (x_val_reduced_300D, y_val))

