
import data
import framework

def getResponse(path=None, tag='ResNet34', checkpoint='./log/best.pt'):
    item = {'path': path}
    model = framework.Model(tag=tag, checkpoint=checkpoint)
    model.eval()
    procedure = data.Procedure(item, inference=True, device='cpu')
    image = procedure.getImage().unsqueeze(0)
    batch = [None, image, None]
    response = model.getPrediction(batch).item()
    return(response)

