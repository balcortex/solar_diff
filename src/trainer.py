import pandas as pd
import os
from typing import Dict, NamedTuple, List, Optional
from tensorflow import keras
import tensorflow as tf
import datetime
import logging

from src.data_manager import (
    get_dataset_filenames,
    update_database,
    dataframe_to_dataset,
)
from src.utils import create_log_dir

DATASET_NAMES = ["amarillo"]


class SplitedCols(NamedTuple):
    inputs: List[str]
    outputs: List[str]


class Trainer:
    def __init__(
        self,
        dataset_name: str,
        keras_model: keras.Model,
        log_dir: str,
        num_outputs: int,
        data_dir: str = "data",
    ) -> None:
        """
        Parameters:
            dataset(str): name of the dataset. One of the following:
                - amarillo
            keras_model: keras uncompiled model.
            log_dir: path of the path to output results in.
            num_outputs: steps of the forecasting task (e.g. 1 = one-step ahead)
            data_dir: path containing the datasets
        """
        assert dataset_name in DATASET_NAMES, f"´{dataset_name}´ dataset not found!"

        update_database(path=data_dir)

        self._dataset_name = dataset_name
        self._model = keras_model
        self._log_dir = log_dir
        self._num_outputs = num_outputs
        self._data_dir = data_dir

        self._img_dir = os.path.join(log_dir, "figures")
        self._filenames = get_dataset_filenames(data_dir, dataset_name)

        self._dataframes = {}
        self._datasets = {}

        self._load_csv()
        self._cols_names = self._split_cols_name(num_outputs=num_outputs)
        self.input_size = len(self._cols_names.inputs)
        self.output_size = len(self._cols_names.outputs)
        self._create_datasets()
        self._prepare_directories()

    def train(self):
        pass

    def test(self):
        pass

    def _load_csv(self) -> None:
        "Load the csv files associated with the chosen database"
        for name, path in self._filenames.items():
            logging.info(f"Reading {path}")
            self._dataframes[name] = pd.read_csv(path)
            self._dataframes[name].drop(columns=["Unnamed: 0"], inplace=True)

    def _prepare_directories(self) -> None:
        "Create the necessary directories for logging results"
        create_log_dir(path=self._log_dir, parent=True)
        create_log_dir(path=self._img_dir)

    def _split_cols_name(self, num_outputs: Optional[int] = None) -> SplitedCols:
        "Return a tuple of lists containing the names of the rows for inputs and outputs"
        all_columns = list(next(iter(self._dataframes.values())).columns)
        inputs = [column for column in all_columns if "x" in column]
        outputs = [column for column in all_columns if "y" in column]
        if num_outputs:
            outputs = outputs[:num_outputs]
        logging.info(f"Input columns ({len(inputs)}): {inputs}")
        logging.info(f"Outputs columns ({len(outputs)}): {outputs}")
        return SplitedCols(inputs, outputs)

    def _create_datasets(self) -> None:
        "Create tf-datasets for training the model"
        for name in self._filenames.keys():
            self._datasets[name] = dataframe_to_dataset(
                df=self._dataframes[name],
                input_size=len(self._cols_names.inputs),
                output_size=len(self._cols_names.outputs),
            )
