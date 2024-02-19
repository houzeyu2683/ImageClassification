
import torch
import os
import torchvision

class Model(torch.nn.Module):

    def __init__(self, tag='ResNet34', checkpoint=None):
        super(Model, self).__init__()
        if(tag=='ResNet34'):
            weight = 'ResNet34_Weights.IMAGENET1K_V1'
            backbone = torchvision.models.resnet34(weights=weight)
            layer = torch.nn.Sequential(
                torch.nn.Sequential(*(list(backbone.children())[:-1])),
                torch.nn.Flatten(1,-1),
                torch.nn.Linear(512, 2)
            )
            pass
        if(tag=='ResNet50'):
            weight = 'ResNet50_Weights.IMAGENET1K_V2'
            backbone = torchvision.models.resnet50(weights=weight)
            layer = torch.nn.Sequential(
                torch.nn.Sequential(*(list(backbone.children())[:-1])),
                torch.nn.Flatten(1,-1),
                torch.nn.Linear(2048, 2)
            )
            pass
        if(tag=='ResNet101'):
            weight = 'ResNet101_Weights.IMAGENET1K_V2'
            backbone = torchvision.models.resnet101(weights=weight)
            layer = torch.nn.Sequential(
                torch.nn.Sequential(*(list(backbone.children())[:-1])),
                torch.nn.Flatten(1,-1),
                torch.nn.Linear(2048, 2)
            )
            pass
        if(tag=='ResNet152'):
            weight = 'ResNet152_Weights.IMAGENET1K_V2'
            backbone = torchvision.models.resnet152(weights=weight)
            layer = torch.nn.Sequential(
                torch.nn.Sequential(*(list(backbone.children())[:-1])),
                torch.nn.Flatten(1,-1),
                torch.nn.Linear(2048, 2)
            )
            pass
        criteria = torch.nn.CrossEntropyLoss()
        self.tag = tag
        self.layer = layer
        self.criteria = criteria
        if(checkpoint): 
            self.load_state_dict(torch.load(checkpoint, map_location='cpu'))
            pass
        return

    def getScore(self, batch):
        _, image, _ = batch
        score = self.layer(image)
        score = score
        return(score)

    def getPrediction(self, batch):
        score = self.getScore(batch)
        prediction = score.argmax(dim=1)
        return(prediction)
        
    def getLoss(self, batch):
        _, _, target = batch
        score = self.getScore(batch)
        loss = self.criteria(score, target)
        return(loss)

    @torch.no_grad()
    def getInference(self, batch):
        score = self.getScore(batch)
        prediction = self.getPrediction(batch)
        inference = (score, prediction)
        return(inference)

    def saveWeight(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        return
        
    pass