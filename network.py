from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    x = range(1, len(acc)+1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, 100*acc, 'b', label='Training Acc')
    plt.plot(x, 100*val_acc, 'r', label='Validation Acc')
    plt.title('Training/Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training Loss')
    plt.plot(x, val_loss, 'r', label='Validation Loss')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# make sure you have cleaned the data first
df = pd.read_csv('cleaned_data.csv')

embedding_dim = 5 # character embedding, so a smaller dimension seems appropriate
# steps:
# convert each word into a vector that can be input into the embedding layer, with each character
# as a separate vector. We will use one hot encoding to do this.
chars = sorted(list(set(''.join([word for word in df['word']]))))
char_set_size = len(chars)
max_word_len = max([len(word) for word in df['word']])
char_map = {c:(i+1) for i, c in enumerate(chars)}
num_langs = len(set(df['lg_id']))

words = df['word'].values
lg_ids = df['lg_id'].values

x_train, x_test, y_train, y_test = train_test_split(
    words,
    lg_ids,
    test_size=0.2,
    random_state=1024)

y_train = to_categorical(y_train, num_langs)
y_test = to_categorical(y_test, num_langs)

# need to make each word into a vector, e.g. 'abc' -> [1, 2, 3, 0, ..., 0]
# padded with zeros so all vectors are the same length
# can probably do it efficiently, but ill just use a for loop

x = []
for x_vec in (x_train, x_test):
    mapped = [[char_map[c] for c in word] for word in x_vec]
    mapped = np.array(pad_sequences(mapped, padding='post', maxlen=max_word_len))
    x.append(mapped)
x_train, x_test = x

def ConvolutionalModel():
    model = Sequential()
    model.add(layers.Embedding(
        input_dim=char_set_size+1,
        output_dim=embedding_dim,
        input_length=max_word_len))
    # output is (batch_size, 18, 5) or (batch_size, max_word_len, embedding_dim)
    # note about convolutions: a 1D convolutional layer can still go over 2D inputs
    # it just covers the entire last dimension. eg the kernel sizes for this ex
    # would be something like 2x5 or 3x5, always Nx5, since 5 is the last dimension
    # in this case.
    model.add(layers.Conv1D(
        filters=50,
        kernel_size=3,
        input_shape=(max_word_len, embedding_dim),
        padding='valid',
        activation='relu'))
    '''
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 18, 5)             195       
    _________________________________________________________________
    conv1d (Conv1D)              (None, 16, 12)            192       
    =================================================================
    Total params: 387
    Trainable params: 387
    Non-trainable params: 0
    _________________________________________________________________
    '''
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(
        pool_size=2,
        strides=1,
        padding='same'))
    model.add(layers.Conv1D(
        filters=25,
        kernel_size=2,
        padding='valid',
        activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(
        pool_size=2,
        strides=1,
        padding='same'))
    # need to flatten before sending to fc layer
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=50, 
        activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(
        units=num_langs,
        activation='softmax'))

    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics='accuracy')

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        verbose=1,
        validation_data=(x_test, y_test),
        batch_size=300)

    return model, history

def FullyConnectedModel():
    model = Sequential()
    model.add(layers.Embedding(
        input_dim=char_set_size+1,
        output_dim=embedding_dim,
        input_length=max_word_len))
    # if my understanding is correct, this will basically negate
    # any advantages of the embedding layer, so is totally dumb
    # but i will try it anyway!
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=20,
        activation='relu'))
    model.add(layers.Dense(
        units=num_langs,
        activation='softmax'))
    
    model.summary()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics='accuracy')

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        verbose=1,
        validation_data=(x_test, y_test),
        batch_size=500)

    return model, history

def NoEmbeddingModel():
    model = Sequential()
    model.add(layers.Dense(
        input_shape=(18,),
        units=100,
        activation='relu'))
    model.add(layers.Dense(
        units=50,
        activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(
        units=num_langs,
        activation='softmax'))

    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        verbose=1,
        validation_data=(x_test, y_test),
        batch_size=500)

    return model, history

def MergedConvolutionalModel():
    # we want to apply multiple filter types to the original inputs
    # with kernels in (e.g.) the range (2, 3, 4, 5) and some with dilation
    # in (also e.g.) (1, 2, 3) to capture finer details of the placement
    # of character in the word. In order to do this, we must use multiple
    # sublayers and combine them together using keras' Merge layer. I will
    # use the concatenate option to not lose any data.

    # we use the functional api here
    embedding_input = keras.Input(shape=(18,), name='word')
    embedding_output = layers.Embedding(
        input_dim=char_set_size+1,
        output_dim=embedding_dim,
        input_length=max_word_len,
    )(embedding_input)

    conv_layers = []
    for kernel_size in (2, 3, 4):
        for dilation in (1, 2):
            x = layers.Conv1D(
                filters=10,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                activation='relu',
                padding='valid',
            )(embedding_output)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(
                pool_size=2,
                strides=2,
                padding='valid',
            )(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(
                units=8,
                activation='relu',
            )(x)
            
            conv_layers.append(x)

    x = layers.concatenate(conv_layers, axis=1)
    #x = layers.Dropout(0.2)(x)
    #x = layers.Dense(
    #    units=100,
    #    activation='relu',
    #)(x)
    x = layers.Dense(
        units=100,
        activation='relu',
    )(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=num_langs,
        activation='softmax',
    )(x)

    model = keras.Model(
        inputs=embedding_input,
        outputs=x,
    )

    keras.utils.plot_model(
        model,
        'merged.png',
        show_shapes=True,
    )

    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    input('''
            Model compiled successfully.
            Graph topology in merged.png.
            Press any key to train the model.
    ''')

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        verbose=1,
        validation_data=(x_test, y_test),
        batch_size=300
    )

    return model, history


plot_history(MergedConvolutionalModel()[1])

#plot_history(ConvolutionalModel())
