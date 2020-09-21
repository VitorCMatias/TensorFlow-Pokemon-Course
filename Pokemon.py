import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('./dataset/datasets_635_1677_pokemon_alopez247.csv')
df = df[
    ['isLegendary', 'Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Color',
     'Egg_Group_1', 'Height_m', 'Weight_kg', 'Body_Style']]
df['isLegendary'] = df['isLegendary'].astype(int)


def dummy_creation(dataframe, dummy_categories):
    """
    This function first uses pd.get_dummies to create a dummy DataFrame of that category. As it's a separate
    DataFrame, we'll need to concatenate it to our original DataFrame. And since we now have the variables
    represented properly as separate columns, we drop the original column.
    """
    for i in dummy_categories:
        df_dummy = pd.get_dummies(dataframe[i])
        dataframe = pd.concat([dataframe, df_dummy], axis=1)
        dataframe = dataframe.drop(i, axis=1)
    return dataframe


df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color', 'Type_1', 'Type_2'])


def train_test_splitter(dataframe, column):
    dataframe_train = dataframe.loc[df[column] != 1]
    dataframe_test = dataframe.loc[df[column] == 1]

    dataframe_train = dataframe_train.drop(column, axis=1)
    dataframe_test = dataframe_test.drop(column, axis=1)

    return dataframe_train, dataframe_test


df_train, df_test = train_test_splitter(df, 'Generation')


def label_delineator(df_train, df_test, label):
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label, axis=1).values
    test_labels = df_test[label].values

    return train_data, train_labels, test_data, test_labels


train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')


def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return train_data, test_data


train_data, test_data = data_normalizer(train_data, test_data)

# Now we can get to the machine learning! Let's create the model using Keras. Keras is an API for Tensorflow. We have
# a few options for doing this, but we'll keep it simple for now. A model is built upon layers. We'll add two fully
# connected neural layers. The number associated with the layer is the number of neurons in it. The first layer we'll
# use is a 'ReLU' (Rectified Linear Unit)' activation function. Since this is also the first layer, we need to
# specify input_size, which is the shape of an entry in our dataset. After that, we'll finish with a softmax layer.
# Softmax is a type of logistic regression done for situations with multiple cases, like our 2 possible groups:
# 'Legendary' and 'Not Legendary'. With this we delineate the possible identities of the Pokémon into 2 probability
# groups corresponding to the possible labels:


length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=400)


loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print(f'Our test accuracy was {accuracy_value}')


def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
    else:
        print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
        return prediction