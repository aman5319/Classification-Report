import numpy as np


__all__ = ["convert_prob_to_label"]


def convert_prob_to_label(data: np.ndarray) -> np.ndarray:
    """Convert Probability to labels

    Args:
        data: (np.ndarray): Probability output of softmax. The size of data (batch_size, num_classes)

    Returns:
        np.ndarray: Returns pobability converted to labels by finding maximum in last dimension. The output shape is (batch_size,)

    """
    return np.argmax(data, axis=-1)
