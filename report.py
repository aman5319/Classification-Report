from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import numpy as np
import torch
import copy
from utils import convert_prob_to_label
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


class Report:
    def __init__(self, writer=None, classes=None):
        self.writer = SummaryWriter() if writer is None else writer
        self.counter = 0
        self.train_type = ["train", "valid"]
        self.classes = classes
        self.clean_flag = True

    def write_a_batch(self, loss, batch_size, actual, prediction, train_type="train"):
        if self.clean_flag:
            self.init_data_storage()
        assert train_type in self.train_type, f"Train type {train_type} not in the list {self.train_type}"
        self.update_data_iter(batch_size, train_type)
        self.update_loss(loss, batch_size, train_type)
        self.update_actual_prediction(actual, prediction, train_type)
        return self

    def update_actual_prediction(self, actual, prediction, train_type):
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

    def update_loss(self, loss, batch_size, train_type):
        self.loss_count.update({train_type: self.change_data_type(loss, "f")})
        return self

    def update_data_iter(self, batch_size, train_type):
        self.data_count.update({train_type: batch_size})
        self.iter_count.update({train_type: 1})
        return self

    def plot_an_epoch(self, detail=False):
        self.clean_flag = True
        self.counter += 1
        if not detail:
            self.write_to_tensorboard()
        return self

    def init_data_storage(self,):
        self.clean_flag = False
        self.loss_count = Counter(dict(zip(self.train_type, [0] * len(self.train_type))))
        self.data_count = copy.deepcopy(self.loss_count)
        self.iter_count = copy.deepcopy(self.loss_count)
        self.act_pred_dict = dict(zip(self.train_type, [copy.deepcopy(dict()) for i in self.train_type]))

    def change_data_type(self, data, required_data_type):
        if required_data_type == "np" and isinstance(data, torch.Tensor):
            return data.clone().detach().cpu().numpy()
        if required_data_type == "f":
            return data[0] if isinstance(data, np.ndarray) else data.item()
        return data

    def close(self):
        self.writer.close()

    def write_to_tensorboard(self):
        self.plot_loss()
        self.plot_confusion_matrix(self.counter)

    def plot_loss(self,):
        loss_main_tag = "Loss"
        self.loss_count = {i: j / self.iter_count[i] for i, j in self.loss_count.items()}
        self.writer.add_scalars(loss_main_tag, self.loss_count, self.counter)
        return self

    def plot_model(self, model, data):
        self.writer.add_graph(model, data)
        return self

    def plot_confusion_matrix(self, at_which_epoch, simple=True):
        if self.counter % at_which_epoch == 0:
            for tag, value in self.act_pred_dict.items():
                actual, pred = value["actual"], convert_prob_to_label(value["pred"])
                cm = confusion_matrix(actual, pred)
                if not simple:
                    cm_sum = np.sum(cm, axis=1, keepdims=True)
                    cm_perc = cm / cm_sum * 100
                    annot = np.empty_like(cm).astype(str)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            summ = cm_sum[i][0]
                            c = cm[i, j]
                            p = cm_perc[i, j]
                            annot[i, j] = "0" if c == 0 else f"{c}/{summ}\n{p:.1f}%"
                fig = plt.figure(figsize=(10, 10) if simple else (15, 10))
                sns.heatmap(cm, annot=True if simple else annot,
                                 fmt="d" if simple else "",
                                 linewidth=.5, cmap="YlGnBu", linecolor="Black",
                                 figure=fig,
                                 xticklabels=self.classes, yticklabels=self.classes)
                self.writer.add_figure(f"Confusion Matrix/{tag}", fig, global_step=self.counter)
        return self

    def plot_precision_recall(self):
        if all(["train" in self.train_type, "valid" in self.train_type]):
            actual, pred = self.act_pred_dict["valid"]["actual"], convert_prob_to_label(self.act_pred_dict["valid"]["pred"])
            output_valid = classification_report(actual, pred, output_dict=True)
            actual, pred = self.act_pred_dict["train"]["actual"], convert_prob_to_label(self.act_pred_dict["train"]["pred"])
            output_train = classification_report(actual, pred, output_dict=True)
            for key in output_train:
                if isinstance(output_train[key], dict):
                    for key1 in output_train[key]:
                        if key1 != "support":
                            scaler_tag = {"train": output_train[key][key1], "valid": output_valid[key][key1]}
                            self.writer.add_scalars(f"{key}/{key1}", scaler_tag, self.counter)
                else:
                    scaler_tag = {"train": output_train[key], "valid": output_valid[key]}
                    self.writer.add_scalars(key, scaler_tag, self.counter)
        return self
