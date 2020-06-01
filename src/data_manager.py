import pandas as pd
from typing import NamedTuple

from __future__ import annotations


class Dataset(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame


class DataManager:
    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path

        self._train = None
        self._test = None

    def load(self) -> DataManager:
        "Load and process the csv file containing the database"
        self._train = pd.load_csv(self._csv_path)
        # We don't need the first column (index)
        self._train.drop(columns=["Unnamed: 0"], inplace=True)

        return self

    def split_train_test(
        self, num_levels: int = 10, seed: int = 42, sort_by_level: bool = True
    ) -> Dataset:
        """Split database into training and test sets.
        The resulting test set contains representative days with different amounts of cumulative irradiance measurements (e.g., extreme cloudy days, cloudy days, sunny days).
        All measurements included in the test set are dropped from the training set.

        Params:
            num_levels: days to be separated into the test set according to the cumulative irradiance
            seed: seed for repetitive results
            sort_by_level: sort the test set from lower cumulative irradiance (False=random order)

        """
        if self._train and not self._test:
            # self._train =
            # self._test =
            pass
        return Dataset(self._train, self._test)
