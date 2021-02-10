# deeplobeai


### 1.installation
`pip install deeplobeai/`

#### The below dependencies installed automatically
- pytorch== 1.4.0
- pytorch-lightning==1.1.7
- PIL == 7.0.0
- torchvision==0.5.0

### 2.Sample classfication code

```
import deeplobeai
from deeplobeai.classification import Classification

cl_model = Classification()
cl_model.load_data('Dataset)
cl_model.train() #cl_model.train(epochs = 10) for more training
cl_model.predict('sampleimage')
```
