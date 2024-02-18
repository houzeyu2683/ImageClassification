
import data
import framework

item = {'path': 'resource/2023壹電視新聞07-09/-6E6ULf_a4k/skip/-6E6ULf_a4k_30_32.jpg'}
model = framework.Model(tag='ResNet34', path='./log/best.pt')
model.eval()
procedure = data.Procedure(item, inference=True, device='cpu')
image = procedure.getImage(application=True)
batch = [None, image, None]
model.getPrediction(batch).item()

