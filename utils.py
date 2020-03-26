import numpy as np
import json
import collections


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
        return str(self.get_dict_repr())

    def save_config_json(self, path):
        with open(str(path), "w") as f:
            json.dump(self.get_dict_repr(), f)
            print("Configuration Saved")

    @classmethod
    def load_config_json(cls, path):
        with open(str(path)) as f:
            d = json.load(f)
            for i in d.keys():
                if isinstance(d[i], dict):
                    d[i] = Config(**d[i])
            print("Saved Configuration Loaded")
            return cls(**d)

    def get_dict_repr(self,):
        d = {**self.__dict__}
        for i in self.__dict__.keys():
            if isinstance(d[i], Config):
                d[i] = d[i].__dict__
        return d


class HyperParameters(Config):

    def __init__(self, **configs):
        self.update(**configs)

    @staticmethod
    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in  d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(HyperParameters.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
