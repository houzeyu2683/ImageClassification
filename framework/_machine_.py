
import torch
import tqdm

def getOptimization(model=None, method='adam', rate=0.0005):
    if(method=='adam'):
        optimization = torch.optim.Adam(
            model.parameters(), 
            lr=rate, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=0, 
            amsgrad=False
        )
        pass
    if(method=='sgd'):
        optimization = torch.optim.SGD(
            model.parameters(), 
            lr=rate, 
            momentum=0, 
            dampening=0,
            weight_decay=0, 
            nesterov=False
        )
        pass
    return(optimization)

def getSchedule(optimization):
    plan = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    schedule = plan(optimization, T_0=20, T_mult=1)
    return(schedule)

class Machine:

    def __init__(self, model=None, device="cpu"):
        self.model  = model
        self.device = device
        return

    def makeLearning(self, engine):
        model = self.model.to(self.device)
        model.train()
        optimization = getOptimization(model, method='adam', rate=0.0005)
        schedule = getSchedule(optimization)
        existence = getattr(self, 'learning', False)
        if(not existence): self.learning = []
        epoch = {'score':[], 'prediction':[], 'target':[]}
        progress = tqdm.tqdm(engine, leave=False)
        for batch in progress:
            _, _, target = batch
            optimization.zero_grad()
            loss = model.getLoss(batch)
            loss.backward()
            optimization.step()
            score, prediction, target = (
                model.getScore(batch)[:,1].detach().cpu().tolist(),
                model.getPrediction(batch).detach().cpu().tolist(),
                target.detach().cpu().flatten().tolist()
            )
            epoch['score'] += score
            epoch['prediction'] += prediction
            epoch['target'] += target
            continue
        schedule.step()
        self.learning += [epoch]
        return

    @torch.no_grad()
    def makeInference(self, engine):
        model = self.model.to(self.device)
        model.eval()
        progress = tqdm.tqdm(engine, leave=False)
        inference = {'score':[], 'prediction':[], 'target':[]}
        for batch in progress:
            _, _, target = batch
            score, prediction, target = (
                model.getScore(batch)[:,1].detach().cpu().tolist(),
                model.getPrediction(batch).detach().cpu().tolist(),
                target.detach().cpu().flatten().tolist()
            )
            inference['score'] += score
            inference['prediction'] += prediction
            inference['target'] += target
            continue
        self.inference = inference
        return
    
    pass

