import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from data import ImageClassificationDataModule
from model import ImageClassificationCNN

data = ImageClassificationDataModule(data_dir="data/usb_frames", batch_size=8)
model = ImageClassificationCNN(num_classes=4)

trainer = L.Trainer(max_epochs=5, callbacks=[ModelCheckpoint(dirpath="checkpoints", monitor="val_loss")])
trainer.fit(model, data)
trainer.test(model, data.test_dataloader())

trainer.save_checkpoint("checkpoints/model.ckpt")