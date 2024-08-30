#!/usr/bin/env python
# coding: utf-8

# In[1]:




import csv
import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('data.tsv', sep='\t')
print(np.array(dataset.shape))
print(np.array(dataset))

dataset.head()

train_Data = dataset.copy()
train_Out = train_Data.pop('Outcome')

# Training Dataset:
train_Data = np.array(train_Data)
train_Data

train_Out

train_Data, test_Data, train_Out, test_Out = train_test_split(train_Data, train_Out, test_size=0.2, random_state=12)

# 7 Layer [Dense - fully connected], First Hidden Layer 800 nodes/units, Output Layer has 1 node
# input shape to it is just 8 value (8 dimentional input - 8 Features).
# activation = "name of the activation function": https://www.tensorflow.org/api_docs/python/tf/keras/activations

data_model = keras.Sequential(
    [
      #  keras.layers.Flatten()
      keras.layers.Dense(units=800, input_dim=8, name='hidden_layer_1'),
      keras.layers.Dense(units=500, name='hidden_layer_2'),
      keras.layers.Dense(units=250, name='hidden_layer_3'),
      keras.layers.Dense(units=150, name='hidden_layer_4'),
      keras.layers.Dense(units=75, name='hidden_layer_5'),
      keras.layers.Dense(units=40, name='hidden_layer_6'),
      keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid, name='output_layer')
    ]
)

print(data_model.summary())

data_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

# Train the NN
data_model.fit(train_Data, train_Out, validation_data=(test_Data, test_Out), epochs=500, verbose=1)

loss, accuracy = data_model.evaluate(test_Data, test_Out, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

print(data_model.predict(test_Data))
print(test_Out)


# In[2]:



# Predict outcomes
predictions = data_model.predict(test_Data)

# Convert predictions to binary outcomes
predicted_outcomes = (predictions > 0.5).astype(int)

# Print predictions and actual outcomes
print("Predictions vs Actual Outcomes:")
for i in range(len(test_Data)):
    print(f"Predicted: {predicted_outcomes[i][0]}, Actual: {test_Out.iloc[i]}")

# Optionally, show a few examples more clearly
results_df = pd.DataFrame({"Predicted": predicted_outcomes.flatten(), "Actual": test_Out.reset_index(drop=True)})
print(results_df.head(10))


# In[ ]:




