import pandas as pd
import os
from typing import Dict
from tensorflow import keras
import datetime

from src.data_manager import get_dataset_filenames


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
        """
        self._dataset_name = dataset_name
        self._log_dir = log_dir

        self._filenames = get_dataset_filenames(data_dir, dataset_name)

        self._prepare_directories()


files = get_dataset_filenames("data", "amarillo")
print(os.path.exists("data"))
print(files)
