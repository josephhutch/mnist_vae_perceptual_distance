import torch
import torchvision.models as models


class VGG16PerceptualMeasurer(torch.nn.Module):
    def __init__(self):
        super(VGG16PerceptualMeasurer, self).__init__()
        model = models.vgg16(pretrained=True)
        self.model = model
        self.features = model.features
        self.avgpool = model.avgpool
        self.feature_list = torch.nn.ModuleList(list(model.features))
        self.classifier = torch.nn.ModuleList(list(model.classifier))
        return

    def forward(self, x):
        hidden = []
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        for ii, model in enumerate(self.classifier):
            x = model(x)
            if ii in {22, 25, 27, 29}:
                hidden.append(x)

        return hidden

    def use_convs(self, x):
        hidden = []
        for ii, model in enumerate(self.feature_list):
            x = model(x)
            if ii in {1, 4, 6}:
                hidden.append(x)

        return hidden




