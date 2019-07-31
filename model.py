# https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/finetuning_torchvision_models_tutorial.md
# https://github.com/microsoft/tensorwatch
# https://github.com/rwightman/pytorch-image-models

# import torch.functional as F
import torch.nn as nn
from torchvision import models
import timm

from torchviz import make_dot
from torchsummary import summary
from facenet_pytorch import InceptionResnetV1


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, feature_extract, use_pretrained=True):
    """Pre-trained model head"""
    model_ft = None
    input_size = 0

    if model_name == 'resnet50':
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == 'mobilenet_v2':
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == 'shufflenet_v2':
        model_ft = models.shufflenet_v2_x1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == 'mixnet':
        model_ft = timm.create_model('tf_mixnet_s', pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == 'facenet':
        model_ft = InceptionResnetV1(pretrained='vggface2')
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 160
    else:
        raise NameError("No such pre-trained model...")

    return model_ft, input_size


class MultiTaskNet(nn.Module):
    """Multitasking learning architecture"""

    def __init__(self, model_name, num_embeddings, feature_extract=True, use_pretrained=True):
        super(MultiTaskNet, self).__init__()
        self.model_base, self.input_size = initialize_model(model_name, feature_extract, use_pretrained)
        assert isinstance(num_embeddings, int)

        if model_name == 'mixnet':
            num_ftrs = self.model_base.classifier.in_features
            self.model_base.classifier = nn.Linear(num_ftrs, num_embeddings)
        elif model_name == 'mobilenet_v2':
            num_ftrs = self.model_base.classifier[1].in_features
            self.model_base.classifier[1] = nn.Linear(num_ftrs, num_embeddings)
        elif model_name == 'facenet':
            num_ftrs = self.model_base.last_linear.in_features
            self.model_base.last_linear = nn.Linear(num_ftrs, num_embeddings)
            self.model_base.last_bn = nn.BatchNorm1d(num_embeddings, eps=0.001, momentum=0.1, affine=True)
        else:
            num_ftrs = self.model_base.fc.in_features
            self.model_base.fc = nn.Linear(num_ftrs, num_embeddings)

        self.face_branch = nn.Sequential(
            nn.Linear(num_embeddings, 128),
            nn.ReLU(True),
            nn.Linear(128, 11)
        )
        self.mouth_branch = nn.Sequential(
            nn.Linear(num_embeddings, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        self.eyebrow_branch = nn.Sequential(
            nn.Linear(num_embeddings, 128),
            nn.ReLU(True),
            nn.Linear(128, 14)
        )
        self.eye_branch = nn.Sequential(
            nn.Linear(num_embeddings, 128),
            nn.ReLU(True),
            nn.Linear(128, 5)
        )
        self.nose_branch = nn.Sequential(
            nn.Linear(num_embeddings, 128),
            nn.ReLU(True),
            nn.Linear(128, 4)
        )
        self.jaw_branch = nn.Sequential(
            nn.Linear(num_embeddings, 128),
            nn.ReLU(True),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.model_base(x)
        x_face = self.face_branch(x)
        x_mouth = self.mouth_branch(x)
        x_eyebrow = self.eyebrow_branch(x)
        x_eye = self.eye_branch(x)
        x_nose = self.nose_branch(x)
        x_jaw = self.jaw_branch(x)

        return x_face, x_mouth, x_eyebrow, x_eye, x_nose, x_jaw


if __name__ == '__main__':
    # model_base, input_size = initialize_model('shufflenet_v2', feature_extract=True)
    # x = torch.randn(1, 3, 224, 224).requires_grad_(True)
    # vis_graph = make_dot(model_base(x), params=dict(list(model_base.named_parameters()) + [('x', x)]))
    # vis_graph.view()
    # summary(model_base, (3, 224, 224))

    model = MultiTaskNet(model_name='facenet', num_embeddings=256)
    # print(model.jaw_branch[0].weight)
    print(model)
