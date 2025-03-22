import hydra
from omegaconf import DictConfig
from fedops.server.app import FLServer
import models
import data_preparation
from hydra.utils import instantiate

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Instantiate the global model
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__

    # Get the global validation loader
    gl_val_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)

    # Get the test evaluation function
    gl_test_torch = models.test_torch()

    # Create FL server instance
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=gl_val_loader,
        test_torch=gl_test_torch
    )

    # Start the FL process
    fl_server.start()


if __name__ == "__main__":
    main()
