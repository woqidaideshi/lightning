import torch

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel, ManualOptimBoringModel
from pytorch_lightning.strategies import BaguaStrategy


def test_async_algorithm(tmpdir):
    print("-------------in bagua test_async_algorithm----------")
    model = BoringModel()
    bagua_strategy = BaguaStrategy(algorithm="async")
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=2,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    for param in model.parameters():
        assert torch.norm(param) < 3


def test_manual_optimization(tmpdir):
    print("-------------in bagua test_manual_optimization----------")
    model = ManualOptimBoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=1,
        strategy="bagua",
        accelerator="gpu",
        devices=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model)


if __name__ == "__main__":
    model = BoringModel()
    trainer = Trainer(
        strategy=BaguaStrategy(algorithm="gradient_allreduce"),
        accelerator="gpu",
        devices=2,
        max_epochs=5,
    )
    trainer.fit(model)
    test_async_algorithm("./")
    test_manual_optimization("./")
