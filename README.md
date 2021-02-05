# deeplobeai


### installation
`pip install deeplobeai/`


Sample classfication code

```
import deeplobeai
import torch
import pytorch_lightning as pl
from deeplobeai import classification

labels,classes = classification.process_data('Test') ###path to the dataset
### data set should have folder inside with class names

data = classification.load_data('Test',batch_size = 16)
data.prepare_data()
data.setup()

from torchvision import models
network = models.resnet18(pretrained = True)
model = classification.build_model(network,'resnet',classes)

criterion = torch.nn.CrossEntropyLoss()
model = classification.Classficationmodel(model,criterion,'adam',lr = 1e-3)

trainer = pl.Trainer(max_epochs=1)
trainer.fit(model,data)
trainer.test()
```
