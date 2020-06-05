from src.data_manager import create_dataset, update_database
from src.trainer import Trainer
from tensorflow import keras
import logging

# create_dataset("data")
# update_database("data")

logging.basicConfig(level=logging.INFO)

a = Trainer("amarillo", "logs/log", num_outputs=1)
# model = get_MLP(a.input_size, a.output_size, (100, 100))
# model.compile(
#     optimizer=keras.optimizers.Adam(lr=1e-3),
#     loss=keras.losses.mse,
#     metrics=[keras.metrics.mse],
# )
# a.set_model(model)
a.fit(epochs=1_000, verbose=0)

# test = a._datasets["test"]
# model.fit(test, epochs=3)
