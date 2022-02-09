import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

from keras_tuner import RandomSearch

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from scipy import signal



logging.basicConfig(filename="simpleNN.log",
                            filemode='a',
                            format=' %(asctime)s  %(message)s',
                            
                            level=logging.DEBUG)

data_position = sys.argv[1]
cs = sys.argv[2]
df = pd.read_csv(f"./cs_{data_position}_{cs}.csv")


print(df.shape)
n_coff_decimate = int(input("enter n_coff_decimate: "))
resamle=signal.decimate(df,n_coff_decimate)
df_orgin= pd.DataFrame(resamle, columns = range(len(resamle[0])))
l,s=df_orgin.shape
print(l,s)
y = pd.DataFrame(index=range(l))
y.reset_index(drop=True,inplace=True)

y["x"] = pd.read_csv("./target/x_str.txt",header=None)
y["y"] = pd.read_csv("./target/y_str.txt",header=None)
y["z"] = pd.read_csv("./target/z_str.txt",header=None)
X_train, X_test, y_train, y_test = train_test_split( df_orgin, y, test_size=0.20)




"""X_train=X_train.iloc[:,:s].values
X_test=X_test.iloc[:,:s].values

X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
im_shape=(X_train.shape[1],1)"""


def build_model(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Int("ann_layers", min_value=1 , max_value=20 , step =1 )):
      model.add(
       layers.Dense(hp.Int("unit_"+str(i), min_value=3 , max_value=512 , step =2 ),
                   activation="relu")
      )
    model.add(layers.Dense(3,activation="linear"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4,5e-3])),
        loss="mse",
        metrics=["accuracy"],
    )
    return model

tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=30,
    executions_per_trial=2,
    overwrite=False,
    directory="ANN",
    project_name="ANN_tuning",
)

tuner.search(X_train, y_train, 
        epochs=100, 
        callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=7)],
        validation_data=(X_test, y_test))
best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)
models_quarter = tuner.get_best_models(num_models=2)

logging.info(f"\n CS_{data_position}_{cs} SimpleNN: {best_hps.values } \n dataset shape: {df.shape} \n Mean squared error: {models_quarter[0].evaluate(X_test,y_test)} \n samples: {s} \n --------------------------------- \n")