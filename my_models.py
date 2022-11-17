import torch
from const import DEVICE

# imports for models 
from torchvision.models.detection import fasterrcnn_resnet50_fpn as FRCNN
from transformers import DetrForObjectDetection as Detr
from models.retinanet import model
from models.ssd300.model import SSD, ResNet


class Model(torch.nn.Module):
    def __init__(self, model_name: str, **args) -> None:
        super().__init__()
        
        if model_name == 'frcnn':
            if 'detections' not in args: args['detections'] = 1000

            self.model = FRCNN(pretrained=True, box_score_thresh=0, 
                    box_detections_per_img=args['detections']).to(DEVICE)
        
        if model_name == 'retinanet':
            if 'model_dir' not in args: args['model_dir'] = 'models'
            if 'state_dir' not in args: 
                args['state'] = 'models/coco_resnet_50_map_0_335_state_dict.pt'

            self.model = model.resnet50(num_classes=80, pretrained=True, 
                    model_dir=args['model_dir']).to(DEVICE)
            self.model.load_state_dict(torch.load(args['state']))
            self.model.freeze_bn()    # freeze batch-norm layers
        
        if model_name == 'detr':
            self.model = Detr.from_pretrained('facebook/detr-resnet-50').to(DEVICE)

        if model_name == 'ssd300':
            if 'state' not in args:
                args['state'] = 'models/SSD300.pth'

            self.model = SSD(backbone=ResNet(), num_classes=81).to(DEVICE)
            checkpoint = torch.load(args['state'])
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.training = False
        self.model.eval()

    def forward(self, data):
        return self.model(data)