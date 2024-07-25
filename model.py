import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from tensorflow.keras import models, layers,Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D,BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Dense,Add, Multiply, Concatenate,Dropout
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Conv2DTranspose as Conv2DT

IMG_SIZE = 256
start_neurons=256

#%% Optimization of Loss Function


import numpy as np

# Define the objective function (minimization problem)
def loss_func(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Mayfly Optimization
def mayfly_optimization(iterations, population_size, search_space):
    # Initialize Mayfly Optimization parameters
    # You'll need to define these based on the algorithm specifics
    population = np.random.uniform(search_space[0], search_space[1], (population_size,2))
    for _ in range(iterations):
        # Implement Mayfly Optimization steps here
        # Update best_solution if a better solution is found
        fitness_values = [loss_func(x,y) for x,y in population]
        # Find the index of the mayfly with the best fitness
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        best_fitness = fitness_values[best_index]

        # Update the position of each mayfly based on the best solution
        for i in range(population_size):
            if i != best_index:
                alpha = np.random.uniform(0.0, 1.0)
                population[i] = population[i] + alpha * (best_solution - population[i])

    return best_solution, best_fitness


# Moth Flame Optimization
def moth_flame_optimization(iterations, population_size, search_space, flame_size=10.0, alpha=0.1):
    # Initialize Moth Flame Optimization parameters
    # You'll need to define these based on the algorithm specifics
    population = np.random.uniform(search_space[0], search_space[1], (population_size,2))

    for iteration in range(iterations):
        # Implement Moth Flame Optimization steps here
        # Update best_solution if a better solution is found

        # Evaluate the fitness of each moth in the population
        fitness_values = [loss_func(x,y) for x,y in population]
        
        # Find the index of the moth with the best fitness
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        best_fitness = fitness_values[best_index]
        
        # Update the position of each moth based on the flame
        for i in range(population_size):
            if i != best_index:
                distance_to_best = np.abs(population[i] - best_solution)
                b = 1.0 - (iteration / iterations)  # B decreases linearly from 1 to 0
                
                # Calculate the new position of the moth
                population[i] = best_solution - distance_to_best #* np.exp(-alpha * b) * (population[i] - best_solution)

    return best_solution, best_fitness



def pr_loss_fun():
    num_iterations = 100  # Number of iterations
    population_size = 20  # Size of the mayfly population
    search_space = (0, 1)  # Search space (adjust as needed)

    mf_best_solution = mayfly_optimization(num_iterations, population_size, search_space)
    mff_best_solution = moth_flame_optimization(num_iterations, population_size, search_space)

    mf_best_solution, mf_best_fitness = tf.cast(mf_best_solution[0], tf.float32)
    mff_best_solution, mff_best_fitness = tf.cast(mff_best_solution[0], tf.float32)
    print(mf_best_solution)
    print(mf_best_fitness)
    
    print(mff_best_solution)
    print(mff_best_fitness)
    
    if loss_func(mf_best_solution, mf_best_fitness) < loss_func(mff_best_solution, mff_best_fitness):
        return mf_best_solution, mf_best_fitness
    else:
        return mff_best_solution, mff_best_fitness
    

#%% Proposed Model


# Data agumentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
    ])


# decomposing using residual layer model
decomp_model = tf.keras.applications.resnet.ResNet101(
        include_top=False,
        input_shape=(256,256,3))


def proposed_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))    
    
    # augmentation
    x = data_augmentation(inputs)
    
    # Decomposition Residual Network
    x = decomp_model(inputs)
    
    # Unet
    conv1 = Conv2D(start_neurons*4,(1,1), activation='relu', padding='same')(x)
    conv1 = Conv2D(start_neurons*4,(1,1), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    
    conv2 = Conv2D(start_neurons*4,(1,1), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(start_neurons*4,(1,1), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((1,1))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((1,1))(conv3)
    pool3 = Dropout(0.5)(pool3)
    
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((1,1))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    #Middle
    convm = Conv2D(start_neurons * 8, (3,3), activation='relu', padding='same')(pool4)
    convm = Conv2D(start_neurons * 8, (3,3), activation='relu', padding='same')(convm)
    
    #upconv part
    deconv4 = Conv2DTranspose(start_neurons*8,(3,3), strides=(1,1), padding='same')(convm)
    uconv4 = Concatenate()([deconv4, conv4])
    
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons*4,(3,3), strides=(1,1), padding='same')(uconv4)
    uconv3 = Concatenate()([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
    
    deconv2 = Conv2DTranspose(start_neurons*4,(3,3), strides=(1,1), padding='same')(uconv3)
    uconv2 = Concatenate()([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons*4,(1,1), strides=(2,2), padding='same')(uconv2)
    conv1 = tf.keras.layers.Resizing(8,8)(deconv1)
    uconv1 = Concatenate()([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
    
    # Convolutional Layer
    conv_layer = Conv2D(1, (1,1), padding='same', activation='sigmoid')(uconv1)
    conv_layer = Conv2D(2, (2,2), padding='same', activation='sigmoid')(conv_layer)
    
    # Deconvolutional layer
    layr = Conv2DT(32, 3, strides=(3, 3), padding='valid')(conv_layer)
    layr = Conv2DT(32, 3, strides=(3, 3), padding='valid')(layr)
    out = Conv2DT(3, 3, strides=(3, 3), padding='valid')(layr)
    
    proposed_model = tf.keras.Model(inputs=inputs, outputs=out)
    proposed_model.compile(loss='mae',
                           optimizer='adam')
    return proposed_model