import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Input

NUM_INPUTS = 96
NUM_OUTPUTS = 1
VAL_SPLIT = 0.1  # Percentage
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10  # Early stoping
LEARNING_RATE = 1e-4


def make_dataset(df: pd.DataFrame, shuffle: bool = True) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(df.to_numpy())
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(lambda window: (window[:NUM_INPUTS], window[NUM_INPUTS:]))
    dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(3)

    return dataset


def get_compiled_model(
    input_shape: Sequence[int],
    num_outputs: int,
    hidden: Sequence[int] = (100, 100, 100),
    activation: str = "relu",
    lr: float = 1e-3,
    batch_norm: bool = True,
) -> keras.Model:
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape))
    if batch_norm:
        model.add(BatchNormalization())
    for h_units in hidden:
        model.add(Dense(h_units, activation="relu"))
    model.add(Dense(num_outputs))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.mse,
        metrics=[keras.metrics.mse],
    )

    return model


def mse(a: np.array, b: np.array) -> float:
    return np.mean(np.square(a - b))


def mae(a: np.array, b: np.array) -> float:
    return np.mean(np.abs(a - b))


def smape(a: np.array, b: np.array) -> float:
    num = np.abs(b - a)
    den = np.abs(a) + np.abs(b)
    sum = np.where(den == 0, 0, num / den)
    return sum.mean()


def scale_dataframe(
    df: pd.DataFrame, cols: Sequence[str] = None, new_max: int = 1, new_min: int = -1
) -> None:
    cols = cols or df.columns
    for col in cols:
        col_max = df[col].max()
        col_min = df[col].min()
        df[col] = new_min + ((df[col] - col_min) * (new_max - new_min)) / (
            col_max - col_min
        )
        assert df[col].min() == new_min
        assert df[col].max() == new_max


def scale(df: pd.DataFrame, cols=None, new_max=1, new_min=-1):
    cols = cols or df.columns
    for col in cols:
        col_max = df[col].max()
        col_min = df[col].min()
        df[col] = new_min + ((df[col] - col_min) * (new_max - new_min)) / (
            col_max - col_min
        )

    return df


def read_csv(path: str, scale: bool = False) -> pd.DataFrame:
    drop_cols = [f"y{i}" for i in range(NUM_OUTPUTS, 48)]  # drop unnecessary outputs
    drop_cols.append("Unnamed: 0")
    df = pd.read_csv(path)
    df.drop(columns=drop_cols, inplace=True)

    if scale:
        print(f"Scaling {path} . . .")
        input_cols = [f"x{i}" for i in range(96)]
        scale_dataframe(df, input_cols)

    return df


df_train = read_csv("data/amarillo_norm_train.csv", scale=True)
df_test = read_csv("data/amarillo_norm_test.csv", scale=True)
df_train_diff = read_csv("data/amarillo_diff_train.csv", scale=True)
df_test_diff = read_csv("data/amarillo_diff_test.csv", scale=True)
df_sky = read_csv("data/amarillo_sky_test.csv")
sky = df_sky["y0"].values
targets = df_test["y0"].values

len_val = int(VAL_SPLIT * len(df_train))  # reserve validation set
val_ind = df_train.sample(len_val).index.values
df_val = df_train.iloc[val_ind]
df_train.drop(labels=val_ind, inplace=True)
df_val_diff = df_train_diff.iloc[val_ind]
df_train_diff.drop(labels=val_ind, inplace=True)

ds_train = make_dataset(df_train)
ds_val = make_dataset(df_val)
ds_test = make_dataset(df_test, shuffle=False)
ds_train_diff = make_dataset(df_train_diff)
ds_val_diff = make_dataset(df_val_diff)
ds_test_diff = make_dataset(df_test_diff, shuffle=False)

model = get_compiled_model(
    input_shape=(NUM_INPUTS,),
    hidden=(100, 100, 100,),
    num_outputs=NUM_OUTPUTS,
    lr=LEARNING_RATE,
)
model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_val,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )
    ],
)
preds = model.predict(ds_test)
preds = preds.T[0]

model_diff = get_compiled_model(
    input_shape=(NUM_INPUTS,),
    hidden=(100, 100, 100,),
    num_outputs=NUM_OUTPUTS,
    lr=LEARNING_RATE,
)
model_diff.fit(
    ds_train_diff,
    epochs=EPOCHS,
    validation_data=ds_val_diff,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )
    ],
)
preds_diff = model_diff.predict(ds_test_diff)
proj_preds = sky - preds_diff.T[0]

print(f"MSE {mse(targets, preds)}")
print(f"MSE diff {mse(targets, proj_preds)}")
print(f"MAE {mae(targets, preds)}")
print(f"MAE diff {mae(targets, proj_preds)}")
print(f"SMAPE {smape(targets, preds)}")
print(f"SMAPE diff {smape(targets, proj_preds)}")

for index, (target, pred, proj_pred) in enumerate(
    zip(
        targets.reshape(1, 10, 144)[0],
        preds.reshape(1, 10, 144)[0],
        proj_preds.reshape(1, 10, 144)[0],
    )
):
    plt.clf()
    plt.plot(target[:48], "b", label="Targets")
    plt.plot(pred[:48], "r", label="Forecast")
    plt.plot(proj_pred[:48], "g", label="Diff Fcst")
    plt.ylim(-100, 1100)
    plt.legend()
    plt.title(f"Irradiance level {index+1}")
    fig_path = os.path.join("figs2", f"fig_level_{index+1}")
    plt.savefig(fig_path)


# MSE 6461.201225098848
# MSE diff 1781.6147084208612
# MAE 47.053116507455705
# MAE diff 18.331995485446313
# SMAPE 0.6205370468405844
# SMAPE diff 0.5762314170842123
