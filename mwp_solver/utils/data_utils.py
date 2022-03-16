from dataset.abstract_dataset import AbstractDataset
from dataset.single_equation_dataset import SingleEquationDataset
from dataset.multi_equation_dataset import MultiEquationDataset

from dataloader.abstract_dataloader import AbstractDataLoader
from dataloader.single_equation_dataloader import SingleEquationDataLoader
from dataloader.multi_equation_dataloader import MultiEquationDataLoader
from utils.enum_type import TaskType

def create_dataset(config):
    """Create dataset according to config
    Args:
        config (mwptoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.
    Returns:
        Dataset: Constructed dataset.
    """
    task_type = config['task_type'].lower()
    if task_type == TaskType.SingleEquation:
        return SingleEquationDataset(config)
    elif task_type == TaskType.MultiEquation:
        return MultiEquationDataset(config)
    else:
        return AbstractDataset(config)

def create_dataloader(config):
    """Create dataloader according to config
    Args:
        config (mwptoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.
    Returns:
        Dataloader module
    """
    task_type = config['task_type'].lower()
    if task_type == TaskType.SingleEquation:
        return SingleEquationDataLoader
    elif task_type == TaskType.MultiEquation:
        return MultiEquationDataLoader
    else:
        return AbstractDataLoader