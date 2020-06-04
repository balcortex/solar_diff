from src.data_manager import create_dataset, update_database
from src.trainer import Trainer
import logging

# create_dataset("data")
# update_database("data")

logging.basicConfig(level=logging.INFO)

a = Trainer("amarillo", None, "logs/log", num_outputs=1)
