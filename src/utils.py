import os
import datetime
from typing import List
from zipfile import ZipFile, ZIP_DEFLATED


def create_log_dir(path: str, fmt: str = "%Y-%m-%d_%H-%M") -> None:
    today = datetime.datetime.now().strftime(fmt)
    os.makedirs(path + "-" + today, exist_ok=True)


def delete_files(file_paths: List[str]) -> None:
    for path in file_paths:
        os.remove(path)


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


def unzip_all(zip_path: str, unzip_path: str) -> None:
    ZipFile(zip_path).extractall(path=unzip_path)
