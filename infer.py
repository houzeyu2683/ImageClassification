import data
import framework

path='./resource/test.csv'
test = data.Unit(path, batch=6, inference=True, device='cuda')
test.makeEngine()
test.getSample()

checkpoint = './log/best.pt'
model = framework.Model(tag='ResNet34', checkpoint=checkpoint)
machine = framework.Machine(model, device='cuda')

machine.makeInference(test.engine)
metric = framework.Metric(
    machine.inference['score'],
    machine.inference['prediction'],
    machine.inference['target']
)
path = './log/test.txt'
print(f'test/loss:{metric.getLoss()}', file=open(path, 'a'))
print(f'test/accuracy:{metric.getAccuracy()}', file=open(path, 'a'))
print(f'test/area under curve:{metric.getAreaUnderCurve()}', file=open(path, 'a'))