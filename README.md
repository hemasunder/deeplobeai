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
from deeplobeai import classification

cl_obj = classification.CL()
cl_obj.load_data('Dataset)
cl_obj.train()
cl_obj.predict('sampleimg')
```
