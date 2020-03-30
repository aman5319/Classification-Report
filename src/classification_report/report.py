from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import numpy as np
import torch
import copy
from .utils import convert_prob_to_label
from .config import HyperParameters
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
import matplotlib.pyplot as plt
from datetime import datetime
import re
from .return_types import LossType, ActualType, PredictionType, TrainType

__all__ = ["Report"]


class Report:
    """Generating Report for classification Model by tracking Model training and  giving different types of metrics to evaluate the Model.

    For any classification Problem during Model's training it is very important to track Model's Weight Biases and Gradients. After training the important part is the model evaluation where we evaluate the model performance. This Report class simplify the evaluation part where all the evaluation metrics are automatically generated for the model.
    It uses Tensorboard to keep a track of all these.

    Features
        1. Model Weights, Biases and Gradients Tracking using Histogram.
        2. Generating GUI graph  of the entire Model.
        3. Graph of Precision, Recall and F1 Score for all the classes for each epoch.
        4. Graph of Macro Avg and Weighted Avg of Precision, Recall and F1-score for eacg epoch.
        5. Training and Validation Loss tracking for each epoch.
        6. Accuracy and MCC metric tracking at each epoch.
        7. Generating Confusion Matrix after certain number of epoch.
        8. Bar Graph for False Positive and False Negative count for each class.
        9. Scatter Plot for the predicited probabilities.
        10. Hyparameter Tracking.

    """
    def __init__(self, classes: TrainType, dir_name: str = None):
        """

        Args:
            classes: A list of classes.
            dir_name: Directory name where tensorboard logs will be saved.

        """
        logdir = "runs/" + datetime.now().strftime("%d:%m:%Y-%H:%M:%S")
        if dir_name is not None:
            logdir = logdir + f"_{dir_name}"
        self.writer = SummaryWriter(log_dir=logdir, flush_secs=15)
        self.counter = 0  # epoch counter
        self.train_type = ["train", "valid"]
        self.classes = [f"c{i}-{j}" for i, j in enumerate(classes)]
        self.clean_flag = True  # Flag to be used to flush the data module to store new data after the epoch

    def write_a_batch(self, loss: LossType,
                      batch_size: int,
                      actual: ActualType,
                      prediction: PredictionType,
                      train: bool = True):
        """This methods records the batch information during train and val phase.

        During training and validation record the loss, batch actual labels and predicted labels.

        Note:
            For prediction don't pass raw_logits pass softmax output.

        Args:
            loss: The batch loss.
            batch_size: The batch size on which the loss was calculated. The batch_size may change during last iteration so calculate batch_size from data.
            actual: The actual labels.
            prediction: The predicted labels.
            train:  True signifies training mode and False Validation Mode.

        Returns:
            Report class instance.

        """
        if self.clean_flag:
            self.init_data_storage()
        train_type = self.train_type[not train]
        self.update_data_iter(batch_size, train_type)
        self.update_loss(loss, batch_size, train_type)
        self.update_actual_prediction(actual, prediction, train_type)
        return self

    def update_actual_prediction(self, actual: ActualType,
                                 prediction: PredictionType,
                                 train_type: TrainType):
        """Stores actual and predicted labels seperately for training and validation and after every batch call the values are appended.
        Args:
            actual: The actual labels.
            prediction: The predicted labels.
            train_type: The labels belong to `train` or `valid`.

        Returns:
            Report class instances.

        """

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

    def update_loss(self, loss: LossType,
                    batch_size: int,
                    train_type: TrainType):
        """Accumlates loss for every batch seperately for training and validation.

        Args:
            loss: The batch loss.
            batch_size: The batch size on which the loss was calculated. The batch_size may change during last iteration so calculate batch_size from data.
            train_type: The Labels belong to `train` or `valid`.

        Returns:
           Report class instance.

        """
        self.loss_count.update({train_type: self.change_data_type(loss, "f") * batch_size})
        return self

    def update_data_iter(self,
                         batch_size: int,
                         train_type: TrainType):
        """Accumlates the iteration count and data point count for training and validation.

        Args:
            batch_size: The batch size on which the loss was calculated. The batch_size may change during last iteration so calculate batch_size from data.
            train_type: The Labels belong to `train` or `valid`.

        Returns:
           Report class instance.

        """

        self.data_count.update({train_type: batch_size})
        self.iter_count.update({train_type: 1})
        return self

    def plot_an_epoch(self, detail: bool = False):
        """Plot an epoch method simplifies ploting standard things which are needed to be plotted after an epoch completion for granular control use this which `detail` = `False` and call other methods on top of it.

        Args:
            detail: whether to use detail mode or not.

        Returns:
            Report class instance.

        """
        self.clean_flag = True
        self.counter += 1  # One Epoch is completed update the counter by one.
        if not detail:
            self.write_to_tensorboard()
        return self

    def init_data_storage(self,):
        """Clean the data storage units after every epoch."""
        self.clean_flag = False
        self.loss_count = Counter(dict(zip(self.train_type, [0] * len(self.train_type))))
        self.data_count = copy.deepcopy(self.loss_count)
        self.mcc = copy.deepcopy(self.loss_count)
        if getattr(self, "iter_count", None) is None:
            self.iter_count = copy.deepcopy(self.loss_count)
        self.act_pred_dict = dict(zip(self.train_type, [copy.deepcopy(dict()) for i in self.train_type]))

    def change_data_type(self, data: LossType, required_data_type: LossType):
        """Change the data type of input to required data type.

        Args:
            data: Input data type.
            required_data_type: Change the data type to given format, can be either `np` or `f`.

        Returns:
            The data in required data type.

        """
        if required_data_type == "np" and isinstance(data, torch.Tensor):
            return data.clone().detach().cpu().numpy()
        if required_data_type == "f":
            return data[0] if isinstance(data, np.ndarray) else data.item()
        return data

    def close(self):
        """Close the tensorboard writer object."""
        self.writer.close()

    def write_to_tensorboard(self):
        """This methods call various other method which write on tensorboard."""
        self.plot_loss()
        self.plot_confusion_matrix(5)
        self.plot_precision_recall()
        self.plot_missclassification_count(5)
        self.plot_mcc()
        self.plot_pred_prob(at_which_epoch=5)

    def plot_loss(self,):
        """Plots loss at the end of the epoch.

        Returns:
           Report class instance.

        """
        loss_main_tag = "Loss"
        self.loss_count = {i: j / self.data_count[i] for i, j in self.loss_count.items()}
        self.writer.add_scalars(loss_main_tag, self.loss_count, self.counter)
        return self

    def plot_model(self, model: torch.nn.Module, data: torch.Tensor):
        """Plot model graph.

        Args:
            model: The model architecture.
            data: The input to the model.

        Returns:
           Report class instance.

        """
        self.model = model
        self.writer.add_graph(model, data)
        return self

    def plot_confusion_matrix(self, at_which_epoch):
        """Plots confusion matrix.

        Args:
            at_which_epoch: After how many epochs the confusion matrix should be plotted. For example if the model is trained for 10 epochs and you want to plot confusion matrix after every 5 epoch then the input to this method will be 5.

        Returns:
            Report class instance.

        """
        if self.counter % at_which_epoch == 0:
            for tag, value in self.act_pred_dict.items():
                actual, pred = value["actual"], convert_prob_to_label(value["pred"])

                # Generating Confusion Matrix

                cm = confusion_matrix(actual, pred)
                cm_sum = np.sum(cm, axis=1, keepdims=True)
                cm_perc = cm / cm_sum * 100
                annot = np.empty_like(cm).astype(str)

                # Generating Annotation
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        summ = cm_sum[i][0]
                        c = cm[i, j]
                        p = cm_perc[i, j]
                        annot[i, j] = "0" if c == 0 else f"{c}/{summ}\n{p:.1f}%"
                fig = plt.figure(figsize=(15, 8))
                ax = sns.heatmap(cm, annot=annot,
                                 fmt="",
                                 linewidth=.5, cmap="YlGnBu", linecolor="Black",
                                 figure=fig,
                                 xticklabels=self.classes, yticklabels=self.classes)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, fontsize=10)
                self.writer.add_figure(f"Confusion Matrix/{tag}", fig, global_step=self.counter)
        return self

    def plot_precision_recall(self):
        """Plots Precision Recall F1-score graph for all Classes with Weighted Average and Macro Average.

        Returns:
            Report class instance.

        """
        if all(["train" in self.train_type, "valid" in self.train_type]):
            actual, pred = self.act_pred_dict["valid"]["actual"], convert_prob_to_label(self.act_pred_dict["valid"]["pred"])
            output_valid = classification_report(actual, pred, output_dict=True, target_names=self.classes)
            actual, pred = self.act_pred_dict["train"]["actual"], convert_prob_to_label(self.act_pred_dict["train"]["pred"])
            output_train = classification_report(actual, pred, output_dict=True, target_names=self.classes)
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

    def plot_missclassification_count(self, at_which_epoch):
        """Plot Misclassification Count for each class.

        Bar graph for False Positive and False Negative Count.

        Args:
            at_which_epoch: After how many epochs the Misclassification Count should be plotted. For example if the model is trained for 10 epochs and you want to plot this after every 5 epoch then the input to this method will be 5.


        Returns:
            Report class instance.

        """
        if self.counter % at_which_epoch == 0 and "valid" in self.train_type:
            actual, pred = self.act_pred_dict["valid"]["actual"], convert_prob_to_label(self.act_pred_dict["valid"]["pred"])
            valid_fp, valid_fn = self.calculate_fp_fn(actual, pred)
            x = np.arange(len(self.classes))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots(figsize=(8, 6))
            rects1 = ax.bar(x - width / 2, valid_fp, width, label='FP')
            rects2 = ax.bar(x + width / 2, valid_fn, width, label='FN')
            ax.set_ylabel('Count')
            ax.set_title('Count of False Positive and False Negative')
            ax.set_xticks(x)
            ax.set_xticklabels(self.classes, rotation=-45)
            ax.legend()

            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 5),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)
            fig.tight_layout()
            self.writer.add_figure("Misclassification/valid", fig, self.counter)
        return self

    def calculate_fp_fn(self, actual: ActualType, pred: PredictionType):
        """Calculates False Postive and False Negative Count per class.

        Args:
            actual: The actual labels.
            pred: The predicted Labels.

        Returns:
            Report class instance.

        """
        true_sum = np.bincount(actual, minlength=len(self.classes))
        pred_sum = np.bincount(pred, minlength=len(self.classes))
        tp_sum = np.bincount(actual[actual == pred], minlength=len(self.classes))
        fp = (pred_sum - tp_sum)
        fn = (true_sum - tp_sum)
        return fp, fn

    def plot_mcc(self,):
        """Plots Mathews Correlation Coefficient.

        Returns:
           Report class instance.

        """
        if all(["train" in self.train_type, "valid" in self.train_type]):
            actual, pred = self.act_pred_dict["valid"]["actual"], convert_prob_to_label(self.act_pred_dict["valid"]["pred"])
            output_valid = matthews_corrcoef(actual, pred)
            actual, pred = self.act_pred_dict["train"]["actual"], convert_prob_to_label(self.act_pred_dict["train"]["pred"])
            output_train = matthews_corrcoef(actual, pred)
            scalar_tag = {"train": output_train, "valid": output_valid}
            self.mcc.update(scalar_tag)
            self.writer.add_scalars("MCC", scalar_tag, self.counter)
        return self

    def plot_pred_prob(self, at_which_epoch: int):
        """Plots scatter plot for the predicted probabilites for each class.

        Args:
            at_which_epoch: After how many epochs the predicted probabilites should be plotted. For example if the model is trained for 10 epochs and you want to plot this after every 5 epoch then the input to this method will be 5.

        Returns:
           Report class instance.

        """
        if self.counter % at_which_epoch == 0 and "valid" in self.train_type:
            actual, pred = self.act_pred_dict["valid"]["actual"], self.act_pred_dict["valid"]["pred"]
            for index, value in enumerate(self.classes):
                f, ax = plt.subplots(1, 1, figsize=(8, 6))
                temp = np.max(pred, axis=-1)[actual == index]
                ax.set_title(value)
                sns.scatterplot(x=temp, y=[*range(len(temp))], ax=ax)
                self.writer.add_figure(f"Prediction Probability/{value}/valid", f, self.counter)
        return self

    def plot_model_data_grad(self, at_which_iter: int):
        """Plot Histogram and Distribution for each layers of model Weights, Bias and Gradients.

        Args:
            at_which_iter: After how many iteration this should be plotted. The ideal way to plot this to plot after every one-half or one-third of the train_iterator.

        Returns:
            Report class instance.

        Examples::
            >>> report.plot_model_data_grad(at_which_iter = len(train_iterator)/2)

        """
        if self.iter_count["train"] % at_which_iter == 0:
            count = self.iter_count["train"] // at_which_iter
            pattern = re.compile(".weight|.bias")
            for key, value in self.model.named_parameters():
                tag_string = ""
                search = pattern.search(key)
                if search is not None:
                    key2 = search.group(0)
                    key1 = pattern.split(key, maxsplit=1)[0]
                    tag_string = f"{key1}/{key2}"
                else:
                    tag_string = f"{key}"
                self.writer.add_histogram(tag_string, value.clone().detach().cpu().numpy(), count)
                if value.grad is not None:
                    self.writer.add_histogram(tag_string + "/grad", value.grad.clone().detach().cpu().numpy(), count)
        return self

    def plot_hparams(self, config: HyperParameters):
        """Plot Hyper parameters for the model. This method should be called once training is over.

        Args:
            config: Hyperparameter Configs.

        Returns:
            Report class instance.

        """
        d = config.get_dict_repr()
        hparam_dict = HyperParameters.flatten(d)
        hparam_dict = {i: j for i, j in hparam_dict.items() if isinstance(j, (int, float, str, bool,))}
        metric_dict = {"Loss": self.loss_count["valid"]}
        if "valid" in self.mcc:
            metric_dict.update({"MCC": self.mcc["valid"]})
        self.writer.add_hparams(hparam_dict, metric_dict)
        return self
