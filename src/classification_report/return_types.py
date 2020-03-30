from typing import Union, List
import numpy as np
import torch

LossType = Union[float, np.ndarray, torch.Tensor]
ActualType = PredictionType = Union[np.ndarray, torch.Tensor]
TrainType = List[str]
