import numpy as np
import json


def convert_prob_to_label(data: np.ndarray) -> np.ndarray:
    return np.argmax(data, axis=-1)


class Config(object):
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def __setattr__(self, name, value):
        super(Config, self).__setattr__(name, value)

    def __getattr__(self, name):
        return super(Config, self).__getattr__(name)

    def __delattr__(self, name):
        return super(Config, self).__delattr__(name)

    def update(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def append_values(self, **kwargs):
        for attr, values in kwargs.items():
            temp_values = getattr(self, attr, None)
            if temp_values is not None:
                if not isinstance(temp_values, (list, tuple)) and not isinstance(values, (list, tuple)):
                    setattr(self, attr, [temp_values, values])
                elif isinstance(temp_values, (list, tuple)) and isinstance(values, (list, tuple)):
                    getattr(self, attr).extend(values)
                else:
                    getattr(self, attr).append(values)

    def __str__(self,):
        return json.dumps(self.__dict__)

    def save_config(self, path):
        with open(str(path), "w") as f:
            json.dump(self.__dict__, f)
        print("Configuration Saved")

    def load_config(self, path):
        with open(str(path)) as f:
            self.update(**json.load(f))
        print("Saved Configuration Loaded")
