from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from typing import Sequence, Optional


class MLP(Model):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden: Sequence[int],
        hidden_activation: Optional[str] = "relu",
    ):
        super(MLP, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self.blocks = [Input(shape=(input_size,))]
        self.blocks.extend([Dense(i, activation=hidden_activation) for i in hidden])
        self.blocks.append(Dense(output_size))

    def call(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs
