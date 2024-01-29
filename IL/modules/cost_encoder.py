import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize

def get_cost_encoder(name, device):
    if name == 'resnet':
        encoder = ResNet(base_encoder=models.__dict__['resnet50']).to(device)
        encoder.eval()
    else:
        raise NotImplementedError
    return encoder

class ResNet(nn.Module):
    def __init__(self, base_encoder=models.__dict__['resnet50']):
        super(ResNet, self).__init__()
        self.encoder = base_encoder(num_classes=1000, pretrained=True)
        self.img_norm = Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                    std=torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, obs, spacial=True, normalize=True):
        obs = obs[:,-3:]/255.0
        if normalize:
            obs = self.img_norm(obs)
        if not spacial:
            h = self.encoder(obs)
            h = h.view(obs.shape[0], -1)
            return h
        else:
            h = obs
        i = 0
        for m in list(self.encoder.children()):
            i += 1
            if i <= 8:
                h = m(h)
        h = h.view(obs.shape[0], -1)
        return h