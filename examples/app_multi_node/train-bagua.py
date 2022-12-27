from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.strategies import BaguaStrategy

if __name__ == "__main__":
    model = BoringModel()
    trainer = Trainer(
        strategy=BaguaStrategy(algorithm="async"),
        accelerator="gpu",
        # devices=1,
    )
    trainer.fit(model)

