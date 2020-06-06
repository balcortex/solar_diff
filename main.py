from src.data_manager import create_dataset, update_database
from src.trainer import TrainerMLP
from tensorflow import keras
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO)

# create_dataset("data")
# update_database("data")

a = TrainerMLP("amarillo", "logs/log", num_outputs=1)
a.fit(epochs=100)
preds = a.test()
targets = a.get_targets()
# for ar in targets.reshape(10, 144):
#     plt.plot(ar[])
# plt.show()
# day_diff = preds["diff"].reshape((10, 144))[0]
day_norm = preds["norm"].reshape((10, 144))[0]
day_targets = targets.reshape((10, 144))[0]
# plt.plot(day_diff, label='diff')
plt.plot(day_norm, label="norm")
plt.plot(day_targets, label="targets")
plt.show()


# a = TrainerMLP("amarillo", "logs/log", num_outputs=1)
# test = a._dataframes["norm_test"]
# test_arr = test["y0"].values
# for ar in test_arr.reshape(10, 144):
#     plt.plot(ar[-48:])
# plt.show()
