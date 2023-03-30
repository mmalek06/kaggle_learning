from collections import namedtuple

from tensorflow import keras


ModelEntry = namedtuple('ModelEntry', ['loss', 'model'])


class ConstrainedPriorityList:
    def __init__(self, max_elements=10):
        self._data: list[ModelEntry] = []
        self._max: int = max_elements

    def add(self, loss: float, model: keras.models.Model):
        if len(self._data) == 0:
            self._data.append(ModelEntry(loss, model))
        else:
            for idx in range(len(self._data)):
                model_entry = self._data[idx]

                if loss < model_entry.loss:
                    new_model_entry = ModelEntry(loss, model)

                    self._data.insert(idx, new_model_entry)

                    break

        while len(self._data) > self._max:
            self._data.pop()

    def top(self, n: int):
        return self._data[:n]
