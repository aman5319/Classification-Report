from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import numpy as np
import torch
import copy


class Report:
    def __init__(self, writer=None, train_type=None):
        self.writer = SummaryWriter() if writer is None else writer
        self.counter = -1
        self.train_type = ["train", "valid", "test"] if train_type is None else train_type
        self.init_data_storage()

    def write_a_batch(self, loss, batch_size, actual=None, prediction=None, train_type="train"):
        assert train_type in self.train_type, f"Train type {train_type} not in the list {self.train_type}"
        self.update_data_iter(batch_size, train_type)
        self.update_loss(loss, batch_size, train_type)
        self.update_actual_prediction(actual, prediction, batch_size, train_type)
        return self

    def update_actual_prediction(actual, prediction, train_type):
        if actual is None and prediction is None :
            return self
        actual = self.change_data_type(actual, "np")
        pred = self.change_data_type(prediction, "np")
        if "actual" not in self.act_pred_dict[train_type]:
            self.act_pred_dict[train_type]["actual"] = actual
        else:
            self.act_pred_dict[train_type]["actual"] = np.concatenate((self.act_pred_dict[train_type]["actual"], actual))
        if "pred" not in self.act_pred_dict[train_type]:
            self.act_pred_dict[train_type]["pred"] = pred
        else:
            self.act_pred_dict[train_type]["pred"] = np.concatenate((self.act_pred_dict[train_type]["pred"], pred))
        return self

    def update_loss(loss, batch_size, train_type):
        self.loss_count.update({train_type: self.change_data_type(loss, "f") * batch_size})
        return self

    def update_data_iter(batch_size, train_type):
        self.data_count.update({train_type: batch_size})
        self.iter_count.update({train_type: 1})
        return self

    def add_loss(self, *loss_args):
        loss_tag_dict = {key: value for key, value in zip(self.train_type, loss_args)}
        self.writer.add_scalars(self.loss_main_tag, loss_tag_dict, self.counter)
        return self

    def plot_an_epoch(self,):
        self.counter += 1
        self.write_to_tensorboard()
        self.init_data_storage()
        return self

    def init_data_storage(self,):
        self.loss_count = Counter(dict(zip(self.train_type,[0]* len(self.train_type))))
        self.data_count = copy.deepcopy(self.loss_count)
        self.iter_count = copy.deepcopy(self.loss_count)
        self.act_pred_dict = dict(zip(self.train_type,[dict()] * len(self.train_type))) 

    def write_to_tensorboard():
        pass

    def change_data_type(self, data, required_data_type):
        if required_data_type == "np" and isinstance(data, torch.Tensor):
            return torch.numpy(data)
        if required_data_type == "f":
            return data[0] if isinstance(data, np.ndarray) else data.item()

    def close(self):
        self.writer.close()
