from src.data_manager import create_dataset, update_database
from src.trainer import Trainer
from src.network import get_model
from tensorflow import keras
import logging

# create_dataset("data")
# update_database("data")

logging.basicConfig(level=logging.INFO)

a = Trainer("amarillo", None, "logs/log", num_outputs=1)
model = get_model(a.input_size, a.output_size, (100, 100))
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.mse,
    metrics=[keras.metrics.mse],
)

test = a._datasets["test"]
model.fit(test, epochs=3)
