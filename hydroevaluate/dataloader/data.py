# -*- coding: utf-8 -*-


class EvalDataset:
    """This dataset is different from the typical dataset in torch, as it is only used for inference and evaluation."""

    def __init__(self, data_cfgs):
        self.data_cfgs = data_cfgs
        self._load_data()

    def _load_data(self):
        self._read_xc()
        # normalization
        norm_x, norm_c = self._normalize()
        self.x, self.c = self._kill_nan(norm_x, norm_c)

    def _read_xc():
        pass

    def _kill_nan():
        pass

    def _normalize():
        pass

    def denormalize(norm_pred):
        pass


def load_dataset(data_cfgs):
    """Load the dataset with the given parameters

    Parameters
    ----------
    dataset_name : str
        The name of the dataset
    data_cfgs : dict
        The configurations for the dataset

    Returns
    -------
    _type_
        _description_
    """
    return EvalDataset(**data_cfgs)
