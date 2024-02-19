import data
import framework

path='./resource/training.csv'
training = data.Unit(path, batch=32, inference=False, device='cuda')
training.makeEngine()
training.getSample()

path='./resource/validation.csv'
validation = data.Unit(path, batch=6, inference=True, device='cuda')
validation.makeEngine()
validation.getSample()

model = framework.Model(tag='ResNet34', checkpoint=None)
machine = framework.Machine(model, device='cuda')

turn = 20
event = framework.Event(folder='log/', step=0)
for index in range(turn):
    machine.makeLearning(training.engine)
    machine.makeInference(validation.engine)
    #
    metric = framework.Metric(
        machine.learning[-1]['score'],
        machine.learning[-1]['prediction'],
        machine.learning[-1]['target']
    )
    key='training/loss', 'training/accuracy', 'training/area under curve'
    event.makeTrace(key[0], value=metric.getLoss())
    event.makeTrace(key[1], value=metric.getAccuracy())
    event.makeTrace(key[2], value=metric.getAreaUnderCurve())
    #
    metric = framework.Metric(
        machine.inference['score'],
        machine.inference['prediction'],
        machine.inference['target']
    )
    key='validation/loss', 'validation/accuracy', 'validation/area under curve'
    event.makeTrace(key[0], value=metric.getLoss())
    event.makeTrace(key[1], value=metric.getAccuracy())
    event.makeTrace(key[2], value=metric.getAreaUnderCurve())
    #
    tag='validation/area under curve'
    improvement = event.getImprovement(tag=tag, maximum=True)
    if(improvement):
        machine.model.saveWeight(f'{event.folder}/{event.step}.pt')
        machine.model.saveWeight(f'{event.folder}/best.pt')
        pass
    event.saveTrace()
    event.makeStep()
    continue
