import torch
from torchvision.models import AlexNet

class Siamese(object):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def init_network(self,):

        model = AlexNet(pretrained=True)
        self.upconv = torch.nn.Conv2d(in_channels=self.args.i, out_channels=3, kernel_size=3)
        self.backbone = torch.nn.Sequential(*list(model.classifier.children())[:-1]) 
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3),
            torch.nn.ReLU(),
        )

    def forward(self, image):

        


