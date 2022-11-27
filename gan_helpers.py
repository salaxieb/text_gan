import tensorflow as tf
import keras_nlp
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

import matplotlib.pyplot as plt


def pad_sequences(sequence, target_len: int=52, embedding_size: int=300) -> np.array:
    sequence = np.array(sequence)
    if sequence.size == 0:
        # empty array
        current_length = 0
        return np.zeros((target_len, embedding_size))
    elif len(sequence.shape) == 1:
        sequence = np.array([sequence])
        current_length = 1
    else:
        current_length = sequence.shape[0]
        
    if current_length >= target_len:
        return sequence[-target_len:]
    
    # padding = np.random.uniform(size=(target_len - current_length, embedding_size))
    padding = np.zeros((target_len - current_length, embedding_size))
    return np.concatenate((padding, sequence), axis=0)



def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(30, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def give_pe(batch_size, position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # angle_rads = angle_rads[np.newaxis, ...]

    return tf.constant([angle_rads for _ in range(batch_size)])

b, n, d = 3, 52, 200
pos_encoding = give_pe(b, n, d)

print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()



def make_generator(complexity=180, random_dim=100, classes=None, target_len=52, embedding_size=200, verbose=True):
    alpha = 0.2
    complexity = int(complexity)
    
    random_vector = keras.layers.Input(shape=[random_dim])
    labels = keras.layers.Input(shape=[classes])
    
    X = keras.layers.Concatenate()([random_vector, labels])
    
    # layer 1
    X = keras.layers.Dense(
        target_len//2//2*complexity,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Reshape((target_len//2//2, complexity))(X)
    
    #layer 2 (None, 13, complexity)
    X = keras.layers.Conv1DTranspose(
        filters=complexity//2,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    #layer 3 (None, 13, complexity/2)
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=2,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    #layer 4 (None, 26, complexity/2)
    pe = K.tile(give_pe(1, 26, 200), (K.shape(X)[0], 1, 1))
    X = keras.layers.Concatenate(axis=-1)([X, pe])
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=2,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    
    #layer 5 (None, 52, complexity)
    pe = K.tile(give_pe(1, 52, 200), (K.shape(X)[0], 1, 1))
    X = keras.layers.Concatenate(axis=-1)([X, pe])
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)

    
    #layer 6 (None, 52, complexity)
    pe = K.tile(give_pe(1, 52, 200), (K.shape(X)[0], 1, 1))
    X = keras.layers.Concatenate(axis=-1)([X, pe])
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    
    #layer 6 (None, 52, complexity)
    pe = K.tile(give_pe(1, 52, 200), (K.shape(X)[0], 1, 1))
    X = keras.layers.Concatenate(axis=-1)([X, pe])
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    
    #layer 6 (None, 52, complexity)
    pe = K.tile(give_pe(1, 52, 200), (K.shape(X)[0], 1, 1))
    X = keras.layers.Concatenate(axis=-1)([X, pe])
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    
    
    #layer 7 (None, 52, complexity)
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    X = keras.layers.Dense(embedding_size)(X)
    
    model = keras.models.Model(inputs=[random_vector, labels], outputs=X)
    model.compile(
        loss='binary_crossentropy',
        metrics=['acc'],
        optimizer='sgd'
    )
    if verbose:
        model.summary()
    return model


def make_generator(complexity=180, random_dim=100, classes=None, target_len=52, embedding_size=200, verbose=True):
    alpha = 0.2
    complexity = int(complexity)
    
    random_vector = keras.layers.Input(shape=[random_dim])
    labels = keras.layers.Input(shape=[classes])
    
    X = keras.layers.Concatenate()([random_vector, labels])
    
    # layer 1
    X = keras.layers.Dense(
        target_len//2//2*complexity,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Reshape((target_len//2//2, complexity))(X)
    
    #layer 2 (None, 13, complexity)
    X = keras.layers.Conv1DTranspose(
        filters=complexity//2,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    #layer 3 (None, 13, complexity/2)
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=2,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    #layer 4 (None, 26, complexity/2)
    pe = K.tile(give_pe(1, 26, 200), (K.shape(X)[0], 1, 1))
    X = keras.layers.Concatenate(axis=-1)([X, pe])
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=2,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    X = keras_nlp.layers.TransformerDecoder(
        intermediate_dim = 200,
        num_heads = 4,
    )(X)
    
    #layer 7 (None, 52, complexity)
    X = keras.layers.Conv1DTranspose(
        filters=complexity,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    
    X = keras.layers.Dense(embedding_size)(X)
    
    model = keras.models.Model(inputs=[random_vector, labels], outputs=X)
    model.compile(
        loss='binary_crossentropy',
        metrics=['acc'],
        optimizer='sgd'
    )
    if verbose:
        model.summary()
    return model



def optimizer():
    return tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)


def make_discriminator(complexity=480, target_len=52, embedding_size=200, classes=None, verbose=True):
    complexity = int(complexity)
    alpha = 0.2
    drop_rate = 0.2
    inp = keras.layers.Input(shape=(target_len, embedding_size))
    X = inp
    
    #layer 1
    X = keras.layers.Conv1D(
        filters=complexity//2,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    #layer 2
    X = keras.layers.Conv1D(
        filters=complexity//3,
        kernel_size=3,
        padding='same',
        strides=2,
    )(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    
    #layer 2
    X = keras.layers.Conv1D(
        filters=complexity//3,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    #layer 2
    X = keras.layers.Conv1D(
        filters=complexity//3,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    
    #layer 2
    X = keras.layers.Conv1D(
        filters=complexity//3,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    
    #layer 2
    X = keras.layers.Conv1D(
        filters=complexity//3,
        kernel_size=3,
        padding='same',
        strides=1,
    )(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    
    #layer 3
    X = keras.layers.Conv1D(
        filters=complexity//4,
        kernel_size=3,
        padding='same',
        strides=2,
    )(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    #layer 4
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(complexity)(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    #layer 5
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(complexity//2)(X)
    X = keras.layers.LeakyReLU(alpha)(X)
    X = keras.layers.Dropout(drop_rate)(X)
    
    # 6 layer
    X_real_fake = keras.layers.Dense(1, activation='sigmoid', name='real_fake')(X)
    # 6 layer
    X_labels = keras.layers.Dense(classes, activation='softmax', name='labels')(X)
    
    model = keras.models.Model(inputs=inp, outputs=[X_real_fake, X_labels], name='discriminator')
    model.compile(loss={'real_fake': 'binary_crossentropy', 'labels': 'categorical_crossentropy'},
                  loss_weights={'real_fake':1, 'labels': 0.01},
                  optimizer=optimizer(),
                  metrics={'real_fake':'acc'})
    if verbose:
        model.summary()
    return model
