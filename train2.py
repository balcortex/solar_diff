import datetime
import logging
import os
import sys
from typing import Optional, Sequence, NamedTuple

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
EPOCHS = 1000
EXPERIMENTS_NUM = 30
PATIENCE = 10  # Early stoping
LEARNING_RATE = 1e-3
BATCH_NORM_LAYER = False
CLIP_PREDS = True
CLIP_INPUTS = False


class Loss(NamedTuple):
    mse: float
    mae: float
    smape: float


class ForecastResult(NamedTuple):
    targets: np.array
    preds: np.array
    loss: Loss


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
    batch_norm: bool = False,
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
    df: pd.DataFrame, cols: Sequence[str] = None, new_max: int = 1, new_min: int = 0
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


def read_csv(path: str, scale: bool = False, **kwargs) -> pd.DataFrame:
    logging.info(f"Reading {path} . . .")
    drop_cols = [f"y{i}" for i in range(NUM_OUTPUTS, 48)]  # drop unnecessary outputs
    drop_cols.append("Unnamed: 0")
    df = pd.read_csv(path)
    df.drop(columns=drop_cols, inplace=True)

    if scale:
        logging.info(f"Scaling {path} . . .")
        input_cols = [f"x{i}" for i in range(96)]
        scale_dataframe(df, input_cols, **kwargs)

    return df


def create_log_dir(path: str, dated: bool = False, fmt: str = "%Y-%m-%d_%H-%M") -> str:
    if dated:
        today = datetime.datetime.now().strftime(fmt)
        path += "-" + today
    os.makedirs(path, exist_ok=True)
    logging.info(f"Folder created {path}")

    return path


def experiment(
    data_model: str,
    scale: bool = False,
    seed: Optional[int] = 42,
    save_figs: bool = False,
    batch_norm_layer: bool = False,
    clip_preds: bool = False,
    clip_inputs: bool = False,
):
    if seed:
        np.random.seed(seed)

    print(f"Running model: {data_model}, scale: {scale}")
    path_fig = create_log_dir(os.path.join("figs", data_model), dated=True)
    path = "data{sep}amarillo_{{model}}_{{dataset}}.csv".format(sep=os.path.sep)

    # mix input differences with real output
    if data_model == "diffmix":
        df_train_norm = read_csv(
            path.format(model="norm", dataset="train"), scale=scale
        )
        df_test_norm = read_csv(path.format(model="norm", dataset="test"), scale=scale)
        df_train_diff = read_csv(
            path.format(model="diff", dataset="train"), scale=scale
        )
        df_test_diff = read_csv(path.format(model="diff", dataset="test"), scale=scale)
        cols = df_train_norm.columns
        x_inputs = [col for col in cols if "x" in col]
        df_train = df_train_diff[x_inputs].copy()
        df_train["y0"] = df_train_norm["y0"].values
        df_test = df_test_diff[x_inputs].copy()
        df_test["y0"] = df_test_norm["y0"].values
        np.array_equal(df_train[x_inputs].values, df_train_diff[x_inputs].values)
        np.array_equal(df_test[x_inputs].values, df_test_diff[x_inputs].values)
        np.array_equal(df_train["y0"].values, df_train_norm["y0"].values)
        np.array_equal(df_test["y0"].values, df_test_norm["y0"].values)
    else:
        df_train = read_csv(path.format(model=data_model, dataset="train"), scale=scale)
        df_test = read_csv(path.format(model=data_model, dataset="test"), scale=scale)

    if clip_inputs and not scale:
        print("Clipping inputs . . .")
        df_train[df_train < 0] = 0
        df_test[df_test < 0] = 0

    # Validation
    len_val = int(VAL_SPLIT * len(df_train))
    val_ind = df_train.sample(len_val).index.values
    df_val = df_train.iloc[val_ind]
    df_train.drop(labels=val_ind, inplace=True)

    # Make datasets
    ds_train = make_dataset(df_train)
    ds_val = make_dataset(df_val)
    ds_test = make_dataset(df_test, shuffle=False)

    model = get_compiled_model(
        input_shape=(NUM_INPUTS,),
        hidden=(100, 100, 100,),
        num_outputs=NUM_OUTPUTS,
        lr=LEARNING_RATE,
        batch_norm=batch_norm_layer,
    )
    print("Training model . . .")
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

    if data_model != "diff":
        targets = df_test["y0"].values
        preds = model.predict(ds_test).T[0]
    # Calculate predictions on original dataset
    else:
        df_sky = read_csv(path.format(model="sky", dataset="test"))
        df_norm = read_csv(path.format(model="norm", dataset="test"))
        preds_temp = model.predict(ds_test).T[0]
        sky = df_sky["y0"].values
        targets = df_norm["y0"].values
        preds = sky - preds_temp

    if clip_preds:
        preds[preds < 0] = 0

    mse_loss = mse(targets, preds)
    mae_loss = mae(targets, preds)
    smape_loss = smape(targets, preds)
    losses = Loss(mse_loss, mae_loss, smape_loss)
    print(losses)

    if save_figs:
        for index, (target, pred) in enumerate(
            zip(targets.reshape(1, 10, 144)[0], preds.reshape(1, 10, 144)[0],)
        ):
            plt.clf()
            plt.plot(target[:48], "b", label="Targets")
            plt.plot(pred[:48], "r", label="Forecast")
            # plt.ylim(-100, 1100)
            plt.legend()
            plt.title(f"Irradiance level {index+1}")
            temp_path = os.path.join(path_fig, f"fig_level_{index+1}")
            plt.savefig(temp_path)
            logging.debug(f"Fig saved to {temp_path}")

        plt.close()

    return ForecastResult(targets, preds, losses)


logging.basicConfig(level=logging.INFO)
exps = {
    "norm_unscaled_exp": {
        "config": {"data_model": "norm", "scale": False,},
        "results": {},
    },
    "diff_unscaled_exp": {
        "config": {"data_model": "diff", "scale": False,},
        "results": {},
    },
    # "norm_scaled_exp": {
    #     "config": {"data_model": "norm", "scale": True,},
    #     "results": {},
    # },
    # "diff_scaled_exp": {
    #     "config": {"data_model": "diff", "scale": True,},
    #     "results": {},
    # },
    # "diffmix_unscaled_exp": {
    #     "config": {"data_model": "diffmix", "scale": False,},
    #     "results": {},
    # },
    # "diffmix_scaled_exp": {
    #     "config": {"data_model": "diffmix", "scale": True,},
    #     "results": {},
    # },
}

for exp_name, exp_dict in exps.items():
    mean_mse = 0
    mean_mae = 0
    mean_smape = 0
    min_mse = np.inf
    max_mse = -np.inf
    exp_min = 0
    exp_max = 0

    # Run experiments
    for i in range(EXPERIMENTS_NUM):
        mdl = exp_dict["config"]["data_model"]
        scl = exp_dict["config"]["scale"]
        current_exp = experiment(
            mdl,
            scale=scl,
            seed=None,
            save_figs=True,
            batch_norm_layer=BATCH_NORM_LAYER,
            clip_preds=CLIP_PREDS,
            clip_inputs=CLIP_INPUTS,
        )
        exps[exp_name]["results"][str(i)] = current_exp

        mean_mse += current_exp.loss.mse
        mean_mae += current_exp.loss.mae
        mean_smape += current_exp.loss.smape

        if current_exp.loss.mse < min_mse:
            min_mse = current_exp.loss.mse
            exp_min = i

        if current_exp.loss.mse > max_mse:
            max_mse = current_exp.loss.mse
            exp_max = i

    # Compute mean errors across experiments
    exps[exp_name]["metrics"] = {
        "mse": mean_mse / EXPERIMENTS_NUM,
        "mae": mean_mae / EXPERIMENTS_NUM,
        "smape": mean_smape / EXPERIMENTS_NUM,
        "min_mse": min_mse,
        "max_mse": max_mse,
        "exp_min": exp_min,
        "exp_max": exp_max,
    }


# Mean preds
for exp_name in exps.keys():
    all_preds = []
    for fcast in exps[exp_name]["results"].values():
        all_preds.append(fcast.preds)
    mean_preds = np.mean(all_preds, axis=0)
    exps[exp_name]["results"]["mean_pred"] = mean_preds
    targets = exps[exp_name]["results"]["0"].targets

    for index, (target, pred) in enumerate(
        zip(targets.reshape(1, 10, 144)[0], mean_preds.reshape(1, 10, 144)[0],)
    ):
        plt.clf()
        plt.plot(target[:48], "b", label="Targets")
        plt.plot(pred[:48], "r", label="Forecast")
        plt.legend()
        plt.title(f"Irradiance level {index+1}")
        dir_path = os.path.join("figs", "means", exp_name)
        os.makedirs(dir_path, exist_ok=True)
        temp_path = os.path.join(dir_path, f"fig_level_{index+1}")
        plt.savefig(temp_path)
        logging.debug(f"Fig saved to {temp_path}")

    plt.close()


for exp_name, exp_dict in exps.items():
    print(f'{exp_name}: {exp_dict["metrics"]}')
