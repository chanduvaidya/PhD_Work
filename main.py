import tensorflow as tf
import matplotlib.pyplot as plt


dataset_path = "dataset"



train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    shuffle=True,
    subset="training",
    validation_split=0.2,
    image_size=(256,256),
    batch_size=32 ,
    seed=123,
    label_mode='categorical'
    )


val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    shuffle=True,
    subset="validation",
    validation_split=0.2,
    image_size=(256,256),
    batch_size=32,
    seed=123,
    label_mode='categorical'
    )


test_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    shuffle=True,
    subset="validation",
    validation_split=0.2,
    image_size=(256,256),
    batch_size=32,
    seed=123,
    label_mode='categorical'
    )


def train(train_ds, val_ds, filepath, epochs=300):
    from model import proposed_model
    import pandas as pd

    pr_model = proposed_model()

    history = pr_model.fit(train_ds,
                           epochs=epochs,
                           validation_data=val_ds,
                           verbose=1
                           )
    pr_model.save('model')



def test(test_ds, filename):
    model = tf.keras.models.load_model('model')
    predict=list()
    for test_batch,_ in test_ds:
        predict.append(model.predict(test_ds))
    import numpy as np
    np.save(filename, np.array(predict))


