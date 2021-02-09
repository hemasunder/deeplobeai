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

data,labels,classes = classification.load_data('Test') 
classification.fit(data,classes,'Test') #model checkpoint saved to Test

classification.predict('Test/12.jpg',labels,'Checkpoint/path/to/file.ckpt')
```
