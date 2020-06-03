from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from typing import Sequence


class MLP(Model):
    def __init__(self, hidden: Sequence[int]):
        super(MLP, self).__init__()
        self.blocks = [Dense(i) for i in hidden]

    def call(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs
