from __future__ import annotations

import os
import random
from typing import Dict, List, NamedTuple, Optional
from zipfile import ZipFile
import tensorflow as tf

import pandas as pd

from src.utils import list_files_extension, delete_files, zip_files, unzip_all


class Dataset(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame


# class DataModel(NamedTuple):
#     real: Dataset
#     diff: Dataset
#     csky: Dataset


class Indices(NamedTuple):
    train_drop: List(int)
    test: List(int)


class DataLoader:
    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path

        self._train = None
        self._test = None
        self._train_indices_drop = None
        self._test_indices = None

    def read_csv(self) -> DataLoader:
        "Load and process the csv file containing the database"
        self._train = pd.read_csv(self._csv_path)
        # We don't need the first column (index)
        self._train.drop(columns=["Unnamed: 0"], inplace=True)

        return self

    def split_train_test(self, train_drop: List[int], test: List[int]) -> DataLoader:
        """Split database into training and test sets.
        The resulting test set contains representative days with different amounts of cumulative irradiance measurements (e.g., extreme cloudy days, cloudy days, sunny days).
        All measurements included in the test set are dropped from the training set.

        Params:
            num_levels: days to be separated into the test set according to the cumulative irradiance
            seed: seed for repetitive results
            sort_by_level: sort the test set from lower cumulative irradiance (False=random order)

        """

        self._test = self._train.iloc[train_drop]
        self._test.reset_index(drop=True, inplace=True)

        self._train.drop(train_drop, inplace=True)
        self._train.reset_index(drop=True, inplace=True)

        return self

    def get_indices(
        self, num_levels: int = 10, seed: int = 42, sort_by_level: bool = True
    ) -> Indices:

        if seed:
            random.seed(seed)

        irr_sum = []
        indices = []

        # Each row contains 144 measurements (3 whole days in total)
        for index in range(0, len(self._train), 144):
            # We only want to compare the last day of the row (i.e. the outputs)
            output_day = self._train.iloc[index].values[-48:]
            indices.append(index)
            irr_sum.append(sum(output_day))

        irr_min = min(irr_sum)
        irr_max = max(irr_sum)
        bin_size = (irr_max - irr_min) / (num_levels - 1)

        # Convert sum to categories
        irr_categorical = [int(i // bin_size) for i in irr_sum]

        # Add randomly one day of each category
        # pylint: disable=unsubscriptable-object
        combined = list(zip(irr_categorical, indices))
        count = 0
        test_levels = []
        test_indices = []
        while count < num_levels:
            samp = random.sample(combined, 1)[0]
            if samp[0] not in test_levels:
                test_levels.append(samp[0])
                test_indices.append(samp[1])
                count += 1

        # Sort from lower to higher level of irradiance
        if sort_by_level:
            test_levels, test_indices = zip(*sorted(zip(test_levels, test_indices)))
            test_levels = list(test_levels)
            test_indices = list(test_indices)

        # Drop every portion of the test set from the training set
        indices_to_drop = []
        for ind in test_indices:
            temp_index = list(range(ind, ind + 144))
            indices_to_drop.extend(temp_index)

        self._train_indices_drop = indices_to_drop
        self._test_indices = test_indices

        return Indices(indices_to_drop, test_indices)

    def get_dataset(self):
        return Dataset(self._train, self._test)

    def save_csv(self, path: str) -> None:
        self._train.to_csv(path + "_train.csv")
        self._test.to_csv(path + "_test.csv")

    def load(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        self._csv_path = None
        self._train = train
        self._test = test


def create_dataset(path: str) -> None:
    print("Creating dataset . . .")

    AMARILLO_PATH = "/home/bal/Dropbox/Solar_Forecasting/texas_amarillo/amarillo_db.csv"
    AMARILLO_DIFF_PATH = (
        "/home/bal/Dropbox/Solar_Forecasting/texas_amarillo/amarillo_db_diff.csv"
    )

    amarillo = DataLoader(AMARILLO_PATH)
    amarillo_diff = DataLoader(AMARILLO_DIFF_PATH)
    amarillo_sky = DataLoader("dummy_path")
    amarillo.read_csv()
    amarillo_diff.read_csv()

    indices = amarillo.get_indices()

    amarillo.split_train_test(indices.train_drop, indices.test)
    amarillo_diff.split_train_test(indices.train_drop, indices.test)

    train_sky = amarillo.get_dataset().train + amarillo_diff.get_dataset().train
    test_sky = amarillo.get_dataset().test + amarillo_diff.get_dataset().test
    amarillo_sky.load(train=train_sky, test=test_sky)

    amarillo.save_csv(os.path.join(path, "amarillo_norm"))
    amarillo_diff.save_csv(os.path.join(path, "amarillo_diff"))
    amarillo_sky.save_csv(os.path.join(path, "amarillo_sky"))

    print("Dataset created.")

    print("Zipping . . .")
    files = list_files_extension(path, extension="csv")
    zip_files(os.path.join(path, "data.zip"), files)


def update_database(path: str) -> None:
    zip_dir = os.path.join(path, "data.zip")
    zip_len = len(ZipFile(zip_dir).infolist())
    dir_len = len(os.listdir(path))

    # discount the zip file
    if not zip_len == (dir_len - 1):
        if dir_len > 1:
            print("Deleting . . .")
            delete_files(list_files_extension(path, extension="csv"))
        print("Unzipping . . .")
        unzip_all(zip_dir, path)
    else:
        print("Data is up to date")


def get_dataset_filenames(dataset_path: str, dataset_name: str) -> Dict[str, str]:
    "Return a dictionary mapping dataset keys to file names"
    fnames = {}

    files = [
        os.path.join(dataset_path, file)
        for file in os.listdir("data")
        if (file.startswith(dataset_name) and file.endswith("csv"))
    ]
    for f in files:
        key = f.split(dataset_name)[1:][0].strip(" _")
        key = key.split(".")[0]
        fnames[key] = f

    return fnames


def dataframe_to_dataset(
    df: pd.DataFrame,
    input_size: int,
    output_size: int,
    batch_size: int = 32,
    buffer_size: int = 1000,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    "Create tf-dataset from pandas-dataframe"
    dataset = tf.data.Dataset.from_tensor_slices(df.to_numpy())
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    dataset = dataset.map(
        lambda window: (
            window[:input_size],
            window[input_size : input_size + output_size],
        )
    )
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False).prefetch(3)

    return dataset


if __name__ == "__main__":
    pass
