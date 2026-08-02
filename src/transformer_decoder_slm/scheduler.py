from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.optim import Optimizer
from .config import Config
from torch.utils.data import DataLoader


class Scheduler:
    def __init__(self, optimizer: Optimizer, config: Config, data_loader: DataLoader):
        self.optimizer = optimizer
        self.config = config

        total_training_steps = len(data_loader) * config.epoch_count
        cosine_steps = total_training_steps - config.warmup_steps

        self.warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=config.minimum_learning_rate,
        )

    def build(self) -> SequentialLR:
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                self.warmup_scheduler,
                self.cosine_scheduler,
            ],
            milestones=[self.config.warmup_steps],
        )
        return scheduler
