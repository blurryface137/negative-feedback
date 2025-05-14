import importlib
from config import NTRSASRecExperimentConfig
from dataset_utils import get_num_items
import torch
from ntrsasrec import NTRSASRec


def load_config(config_file: str) -> NTRSASRecExperimentConfig:
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def build_model(config: NTRSASRecExperimentConfig):
    num_items = get_num_items(config.dataset_name)
    model = NTRSASRec(
        num_items,
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate
    )
    return model


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device
