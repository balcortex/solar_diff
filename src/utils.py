import os
import datetime
from typing import List
from zipfile import ZipFile, ZIP_DEFLATED
import logging


def create_log_dir(path: str, parent=False, fmt: str = "%Y-%m-%d_%H-%M") -> None:
    if parent:
        today = datetime.datetime.now().strftime(fmt)
        path += "-" + today
    os.makedirs(path, exist_ok=True)
    logging.info(f"Folder created {path}")


def delete_files(file_paths: List[str]) -> None:
    for path in file_paths:
        os.remove(path)
        logging.info(f"File removed {path}")


def list_files_extension(directory: str, extension: str) -> List[str]:
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(extension)
    ]


def zip_files(zip_path: str, file_paths: List[str]) -> None:
    with ZipFile(zip_path, "w") as myzip:
        for f in file_paths:
            myzip.write(f, os.path.basename(f), compress_type=ZIP_DEFLATED)
            logging.info(f" Added {f} to {zip_path}")


def unzip_all(zip_path: str, unzip_path: str) -> None:
    ZipFile(zip_path).extractall(path=unzip_path)
    logging.info(f"Extracted all files from {zip_path} to {unzip_path}")
