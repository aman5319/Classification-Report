import json
import collections

__all__ = ["Config", "HyperParameters"]


class Config(object):
    """ This class can be used to create and store data for an type of configuration.

    Args:
        **kwargs: Arbitrary keyword arguments.

    Examples::
        >>> from utils import Config
        >>> training_config = Config(lr=0.1, batch_size=32, device="GPU")
        >>> model_config = Config(number_layers=3, num_head=32)
        >>> model_config.number_layers
        3

    """
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def __setattr__(self, name, value):
        super(Config, self).__setattr__(name, value)

    def __getattr__(self, name):
        return super(Config, self).__getattr__(name)

    def __delattr__(self, name):
        return super(Config, self).__delattr__(name)

    def update(self, **kwargs):
        """To update the config class attributes values.

        This will add new attribute for a non existing attribute in the Config class 
        or replace the value with a new a value for an existing attribute.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Examples::
            >>> training_config = Config(lr=0.1, batch_size=32, device="GPU")
            >>> training_config.lr
            0.1
            >>> training_config.update(lr=0.2,precision:16)
            >>> training_config.lr
            0.2
        """
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def append_values(self, **kwargs):
        """This is method can be used to append values to an existing attribute.

        For Example if using Lr Scheduler then this can be use to track all lr values by appending in a list.
        
        Note:
            The attribute should be prexisiting.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Examples::
            >>> training_config = Config(lr=0.1, batch_size=32, device="GPU")
            >>> training_config.lr
            0.1
            >>> training_config.append_values(lr=0.2)
            >>> training_config.lr
            [0.1,0.2]
            >>> training_config.append_values(lr=[0.3,0.4])
            >>> [0.1,0.2,0.3,0.4]

        """
        for attr, values in kwargs.items():
            temp_values = getattr(self, attr, None)
            if temp_values is not None:
                if not isinstance(temp_values, (list, tuple)) and not isinstance(values, (list, tuple)):
                    setattr(self, attr, [temp_values, values])
                elif isinstance(temp_values, (list, tuple)) and isinstance(values, (list, tuple)):
                    getattr(self, attr).extend(values)
                elif not isinstance(temp_values, (list, tuple)) and isinstance(values, (list, tuple)):
                    temp_l = [temp_values].extend(values)
                    setattr(self, attr, temp_l)
                else:
                    getattr(self, attr).append(values)

    def __str__(self,):
        return str(self.get_dict_repr())

    def save_config_json(self, path):
        """Save the Configuration in Json format.

        Args:
            path: (str) The file path to save json file.

        Examples::
            >>> training_config = Config(lr=0.1, batch_size=32, device="GPU")
            >>> training_config.save_config_json("training_config.json")
            Configuration Saved
        """

        with open(str(path), "w") as f:
            json.dump(self.get_dict_repr(), f)
            print("Configuration Saved")

    @classmethod
    def load_config_json(cls, path):
        """ Loading the saved Configuration.

        Args:
            path: (str) The file from json config will be loaded.

        Returns:
            Config: A Config Class is returned with attributes set from json file

        Examples::
            >>> training_config = Config.load_config_json("training_config.json") # Execute the saving code first.
            >>> training_config.lr
            0.1

        """
        with open(str(path)) as f:
            d = json.load(f)
            for i in d.keys():
                if isinstance(d[i], dict):
                    d[i] = Config(**d[i])  # convert the dictionary into Config object
            print("Saved Configuration Loaded")
            return cls(**d)  # convert the final dictionary into Config Object.

    def get_dict_repr(self,):
        """

        Returns:
            dict: The Dictionary representation of Config class

        Example::
            >>> training_config = Config(lr=0.1, batch_size=32, device="GPU")
            >>> training_config.get_dict_repr()
            {"lr":0.1,"batch_size":32,"device":"GPU"}

        """
        d = {**self.__dict__}
        for i in self.__dict__.keys():
            if isinstance(d[i], Config):
                d[i] = d[i].__dict__
        return d


class HyperParameters(Config):
    """ It stores collections of Config in one place. It inherits the Config Class.

    Args:
        **config: Arbitary Config Objects

    Raises:
        AssertionError: Pass only Config class Object.

    Examples::
        >>> model_config = Config(**{'hid_dim': 512,'n_layers': 8,'n_heads': 8,'pf_dim': 2048,'dropout': 0.1})
        >>> training_config = Config(num_epochs=15, max_lr=0.09, batch_size=64)
        >>> inference_config = Config(batch_size=16)
        >>> hyper = HyperParameters(model_confif = model_config, training_config=training_config,infer_config=inference_config)
        >>> hyper.model_config.hid_dim
        512
        >>> hyper.save_config_json("hyper.json")
        Configuration Saved
        >>> hyper = HyperParameters.load_config_json("hyper.json")
        Saved Configuration Loaded

    """

    def __init__(self, **configs):
        assert all([isinstance(j, Config) for i, j in configs]), "Pass only Config class Object."
        self.update(**configs)

    @staticmethod
    def flatten(d, parent_key='', sep='_'):
        """ Flatten the nested dictionary using Recusrion.

        Args:
            d: (dict) The Dictionary to flatten.
            parent_key: (str) The parent key.
            sep: (str) The sep to be used to join parent_key with nested_key

        Returns:
            dict: Flattened Dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(HyperParameters.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
