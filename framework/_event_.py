
# import traceback
import glob
import os
# import shutil
import pandas
import torch
import torch.utils.tensorboard
import tensorboard.backend.event_processing.event_accumulator

def getBoard(folder):
    board = torch.utils.tensorboard.SummaryWriter(folder)
    return(board)

class Event:

    def __init__(self, folder='./', step=0):
        self.folder = folder
        self.step = step
        return

    def makeStep(self):
        self.step = self.step + 1
        return

    def makeTrace(self, key, value):
        existence = getattr(self, 'board', False)
        if(not existence): 
            self.board = getBoard(self.folder)
            pass
        self.board.add_scalar(key, value, self.step)
        Event = tensorboard.backend.event_processing.event_accumulator
        accumulation = Event.EventAccumulator(self.folder)
        accumulation.Reload()
        tag = accumulation.Tags()["scalars"]
        trace = {}
        for key in tag:
            element = accumulation.Scalars(key)
            step = list(map(lambda x: x.step, element))
            value = list(map(lambda x: x.value, element))
            trace[key] = pandas.DataFrame({"step": step, "value": value})
            continue
        self.trace = trace
        return

    def saveTrace(self):
        for key, value in self.trace.items():
            path = os.path.join(self.folder, f"{key}.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            value.to_csv(path, index=False)
            continue
        return

    def getImprovement(self, tag=None, maximum=True):
        trace = self.trace[tag]
        if(maximum): trace = trace.sort_values(by='value', ascending=False)
        else: trace = trace.sort_values(by='value', ascending=True)
        step = trace.head(1)['step'].item()
        improvement = (step == self.step)
        return(improvement)

    pass