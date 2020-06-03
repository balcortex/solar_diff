import pandas as pd
import os
from typing import Dict
from tensorflow import keras
import datetime

from src.data_manager import get_dataset_filenames, update_database
from src.utils import create_log_dir


class Trainer:
    def __init__(
        self,
        dataset_name: str,
        keras_model: keras.Model,
        log_dir: str,
        data_dir: str = "data",
    ) -> None:
        """
        Parameters:
            dataset(str): name of the dataset. One of the following:
                - amarillo
            keras_model: Keras compiled model.
            log_dir: path of the path to output results in.
            data_dir: path containing the datasets
        """
        self._dataset_name = dataset_name
        self._log_dir = log_dir
        self._img_dir = os.path.join(log_dir, "figures")

        update_database(path=data_dir)

        self._filenames = get_dataset_filenames(data_dir, dataset_name)

        self._prepare_directories()

    def train(self):
        pass

    def test(self):
        pass

    def _prepare_directories(self):
        create_log_dir(path=self._log_dir)
        create_log_dir(path=self._img_dir)
