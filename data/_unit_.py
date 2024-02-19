
import functools
import torch
import torch.utils.data
import torchvision
import PIL.Image
import pandas
import torch.utils.data

def getIteration(path):
    iteration = pandas.read_csv(path).to_dict('index')
    return(iteration)

class Element(torch.utils.data.Dataset):

    def __init__(self, path):
        self.path = path
        return

    def __getitem__(self, index):
        existence = getattr(self, 'iteration', False)
        if(not existence): self.iteration = getIteration(self.path)
        item = self.iteration[index]
        return(item)
    
    def __len__(self):
        existence = getattr(self, 'iteration', False)
        if(not existence): self.iteration = getIteration(self.path)
        length = len(self.iteration)
        return(length)

    pass  

class Unit:

    def __init__(self, path=None, batch=1, inference=False, device='cpu'):
        self.path = path
        self.batch = batch
        self.inference = inference
        self.device = device
        return
    
    def makeEngine(self):
        collation = functools.partial(
            getBatch, 
            inference=self.inference, 
            device=self.device
        )
        element = Element(self.path)
        self.engine = torch.utils.data.DataLoader(
            dataset=element, 
            batch_size=self.batch, 
            shuffle=False if(self.inference) else True, 
            drop_last=False if(self.inference) else True, 
            collate_fn=collation
        )
        return
    
    def getSample(self):
        batch = next(iter(self.engine))
        return(batch)

    pass

def getBatch(iteration=None, inference=False, device='cpu'):
    path, image, target = [], [], []
    for item in iteration:
        procedure = Procedure(
            item=item, 
            inference=inference, 
            device=device
        )
        path += [procedure.getPath()]
        image += [procedure.getImage()]
        target  += [procedure.getTarget()]
        continue
    path = sum(path, [])
    image = torch.stack(image, dim=0)
    target = torch.stack(target, dim=0)
    batch = (path, image, target)
    return(batch)

class Procedure:

    def __init__(self, item=None, inference=False, device='cpu'):
        self.item = item
        self.inference = inference
        self.device = device
        return

    def getPath(self):
        path = [self.item['path']]
        return(path)

    def getImage(self):
        layout    = (256, 256)
        size      = (224, 224)
        mean      = [0.195, 0.195, 0.195]
        deviation = [0.262, 0.262, 0.262]
        degree    = (-10, 10)
        kit = torchvision.transforms
        if(not self.inference):
            transform = kit.Compose([
                kit.RandomHorizontalFlip(p=0.5),
                kit.RandomRotation(degree),
                kit.Resize(layout),
                kit.RandomCrop(size),
                kit.ToTensor(),
                kit.Normalize(mean, deviation),
            ])
            pass
        else:
            transform = kit.Compose([
                kit.Resize(layout),
                kit.CenterCrop(size),
                kit.ToTensor(),
                kit.Normalize(mean, deviation),
            ])
            pass
        image = PIL.Image.open(self.item['path']).convert("RGB")
        image = transform(image).type(torch.FloatTensor)
        image = image.to(self.device)
        # if(application): image = image.unsqueeze(0)
        return(image)

    def getTarget(self):
        target = torch.tensor(self.item['target'])
        target = target.type(torch.LongTensor)
        target = target.to(self.device)
        return(target)

    pass
