import lightning as L

from data import ImageClassificationDataModule
from model import ImageClassificationCNN

data = ImageClassificationDataModule(data_dir="data", batch_size=4)
model = ImageClassificationCNN(num_classes=4)

trainer = L.Trainer(max_epochs=1)
trainer.fit(model, data)