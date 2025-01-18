import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchmetrics import Accuracy, F1Score, MetricCollection
import lightning as L

# use pretrained mobilenetv3_small model


class ImageClassificationCNN(L.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()

        # use pretrained mobilenetv3_small model, adjust the last layer to num_classes
        self.model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.model.classifier[3] = torch.nn.Linear(
            self.model.classifier[3].in_features, num_classes
        )

        # freeze all layers except the last one
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_metrics(logits, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.val_metrics(logits, y)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.test_metrics(logits, y)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
