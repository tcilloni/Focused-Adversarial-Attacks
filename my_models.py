import torch
from const import DEVICE

# imports for models 
from torchvision.models.detection import fasterrcnn_resnet50_fpn as FRCNN
from transformers import DetrForObjectDetection as Detr
from models.retinanet import model as retinanet
from models.ssd300.model import SSD, ResNet


class Model(torch.nn.Module):
    def __init__(self, model_name: str, **args) -> None:
        super().__init__()
        assert model_name in ['retinanet', 'frcnn', 'ssd300', 'detr']
        
        if model_name == 'frcnn':
            if 'detections' not in args: args['detections'] = -1

            self.model = FRCNN(pretrained=True, box_score_thresh=0, 
                    box_detections_per_img=args['detections']).to(DEVICE)
        
        if model_name == 'retinanet':
            if 'model_dir' not in args: args['model_dir'] = 'models'
            if 'state_dir' not in args: 
                args['state'] = 'models/coco_resnet_50_map_0_335_state_dict.pt'

            self.model = retinanet.resnet50(num_classes=80, pretrained=True, 
                    model_dir=args['model_dir']).to(DEVICE)
            self.model.load_state_dict(torch.load(args['state'], map_location=DEVICE))
            self.model.freeze_bn()    # freeze batch-norm layers
        
        if model_name == 'detr':
            self.model = Detr.from_pretrained('facebook/detr-resnet-50').to(DEVICE)

        if model_name == 'ssd300':
            if 'state' not in args:
                args['state'] = 'models/SSD300.pth'

            self.model = SSD(backbone=ResNet(), num_classes=81).to(DEVICE)
            checkpoint = torch.load(args['state'], map_location=DEVICE)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.training = False
        self.model.eval()

    def forward(self, data):
        return self.model(data)
    
    # additional method just for retinanet to avoid any post-processing
    def retinanet_forward_raw(self, img_batch):
        x = self.model.conv1(img_batch)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        features = self.model.fpn([x2, x3, x4])
        regression = torch.cat([
            self.model.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([
            self.model.classificationModel(feature) for feature in features], 
            dim=1)
        anchors = self.model.anchors(img_batch)

        return regression, classification, anchors

    
def detr_scores(model, image):
    out = model(image)
    logits = out['logits'].softmax(-1)[0,:,:-1]
    labels = logits.argmax(1)
    scores = torch.take_along_dim(logits, labels[:,None], 1)
    return scores


# for cleaner code; functions ``f: x -> y`` to get logit predictions from models
forward_fn = {
    'frcnn'    : lambda model, x : model(x)[0]['scores'],
    'detr'     : lambda model, x : detr_scores(model, x)[:,0],
    'retinanet': lambda model, x : model.retinanet_forward_raw(x)[1].softmax(-1)[0],
    'ssd300'   : lambda model, x : model(x.contiguous())[1].softmax(1)[0,1:,:].T,
}