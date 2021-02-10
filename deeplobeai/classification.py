import torch
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms,models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')
from PIL import Image

def process_data(dirr):
    for label in sorted(os.listdir(dirr)):
        print("label",label)
        os.makedirs(os.path.join(dirr,"Train/",label))
        os.makedirs(os.path.join(dirr,"Test/",label))
        os.makedirs(os.path.join(dirr,"Val/",label))
        total = len(os.listdir(os.path.join(dirr,label)))
        val= int(total*0.1)
        print("total",total,"val",val)
        source_dir = os.path.join(dirr,label)
        train_set,test_set,val_set = torch.utils.data.random_split(os.listdir(source_dir),[total-(val+val),val,val])
        print(len(train_set),len(test_set),len(val_set))
        for file in train_set:
            shutil.move(os.path.join(source_dir,file),os.path.join(dirr,"Train/",label))

        for file in test_set:
            shutil.move(os.path.join(source_dir,file),os.path.join(dirr,"Test/",label))

        for file in val_set:
            shutil.move(os.path.join(source_dir,file),os.path.join(dirr,"Val/",label))
        print("Done one",label)
    return sorted(os.listdir(os.path.join(dirr,"Train/"))),len(os.listdir(os.path.join(dirr,"Train/")))
        

class loading_data(pl.LightningDataModule):
    def __init__(self,root_dir):
        super().__init__()
        
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(15),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               ])
        self.batch_size = 16
        self.setup()
        
    def setup(self):
        self.train_set = datasets.ImageFolder(os.path.join(self.root_dir,'Train/'),transform= self.transform)
        self.test_set = datasets.ImageFolder(os.path.join(self.root_dir,'Test/'),transform = self.transform)
        self.val_set = datasets.ImageFolder(os.path.join(self.root_dir,'Val/'),transform = self.transform)
         
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.batch_size, shuffle = True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size= self.batch_size, shuffle = False)
    
    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size= self.batch_size, shuffle = False)

                
def build_model(model,modelname,classes):
    if modelname in ['resnet','inception','googlenet']:
        infeatures = model.fc.in_features
        model.fc = nn.Linear(infeatures,classes)
    if modelname in ['vggnet','alexnet']:
        infeatures = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(infeatures,classes)
    if modelname in ['mobilenet']:
        infeatures = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(infeatures,classes)
    if modelname in ['densenet']:
        infeatures = model.classifier.in_features
        model.classifier = nn.Linear(infeatures,classes)
    return model


class Classificationmodel(pl.LightningModule):
    def __init__(self,classes):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = models.resnet18(pretrained = True)
        self.classes = classes
        self.model = build_model(self.model,'resnet',self.classes)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 0.001
    
    def forward(self,x):
        output = self.model(x)
        return output
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        output = self.forward(x)
        loss = self.criterion(output,y)
        self.log("train_loss",loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y = batch
        output = self.forward(x)
        loss = self.criterion(output,y)
        self.log("val_loss",loss)
    
    def test_step(self,batch,batch_idx):
        x,y = batch
        output = self.forward(x)
        loss = self.criterion(output,y)
        self.log("test_loss",loss)

    def configure_optimizers(self):
        #if self.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(self.parameters(),lr = self.lr)
        #if self.optim.lower() == 'adam':
        #	optimizer = torch.optim.Adam(self.parameters(),lr = self.lr)
        return optimizer

class CL():
    def load_data(self,dataset):
        self.dataset = dataset
        (self.labels,self.classes),self.data = process_data(dataset),loading_data(dataset) 
        print(self.labels, self.classes, self.data)
    
    def train(self):
        model = Classificationmodel(self.classes)
        print("model training started")
        self.checkpoint_callback = ModelCheckpoint(monitor = 'val_loss',dirpath = self.dataset)
        trainer = pl.Trainer(max_epochs = 1,default_root_dir = self.dataset,callbacks = [self.checkpoint_callback])
        trainer.fit(model,self.data)
        print("testing the model now")
        trainer.test()

    def predict(self,img):
        img = Image.open(img).convert('RGB')
        transform = transforms.Compose([transforms.Resize((224,224)),
                           transforms.ToTensor()])
        t_img = transform(img).unsqueeze(0)
        self.path= self.checkpoint_callback.best_model_path
        model = Classificationmodel.load_from_checkpoint(self.path)
        model.eval()
        out = model(t_img)
        _,pred = out.max(1)
        return pred,self.labels[pred.item()]
