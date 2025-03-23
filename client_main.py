#
import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import data_preparation
import models
from fedops.client import client_utils
from fedops.client.app import FLClientTask
import logging
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seeds
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # Show config
    print(OmegaConf.to_yaml(cfg))

    # Load PTB-XL Data
    train_loader, val_loader, test_loader = data_preparation.load_partition(
        data_dir="./dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
        batch_size=cfg.batch_size,
        validation_split=cfg.dataset.validation_split
    )
    logger.info("PTB-XL data loaded")

    # Load model
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__

    # Load training/testing functions
    train_torch = models.train_torch()
    test_torch = models.test_torch()

    # Load local model if available
    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)
    if local_list:
        logger.info('Latest Local Model download')
        model = client_utils.download_local_model(model_type, task_id, listdir=local_list, model=model)

    # Register with FL client
    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": model_name,
        "train_torch": train_torch,
        "test_torch": test_torch
    }

    # Launch FedOps FL Client
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
