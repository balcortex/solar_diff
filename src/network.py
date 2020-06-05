from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from typing import Sequence, Optional


def get_MLP(
    input_size: int,
    output_size: int,
    hidden: Sequence[int],
    hidden_activation: Optional[str] = "relu",
):

    model = Sequential()
    model.add(Input(shape=(input_size,)))
    for units in hidden:
        model.add(Dense(units=units, activation=hidden_activation))
    model.add(Dense(units=output_size))

    return model
