import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorchcv.model_provider import get_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)


def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)


class Head(torch.nn.Module):
    def __init__(self, in_f, out_f, hidden):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, hidden)
        self.m = Mish()
        self.d = nn.Dropout(0.75)
        self.o = nn.Linear(hidden, out_f)
        # self.o = nn.Linear(in_f, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(hidden)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out


class FCN(torch.nn.Module):
    def __init__(self, base, in_f, hidden):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, 1, hidden)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)


net = []

model = get_model("xception", pretrained=False)
model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
model = FCN(model, 2048, 512)
# model = model.cuda()
model.load_state_dict(torch.load("./deepfake-models/model2.pth", map_location=torch.device("cpu")))  # .402
net.append(model)


model = get_model("efficientnet_b1", pretrained=False)
model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
model = FCN(model, 1280, 512)
# model = model.cuda()
model.load_state_dict(torch.load("./deepfake-models/model (.259).pth", map_location=torch.device("cpu")))  # .392
net.append(model)


model = get_model("efficientnet_b1", pretrained=False)
model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
model = FCN(model, 1280, 512)
# model = model.cuda()
model.load_state_dict(torch.load("./deepfake-models/model_16 (.2755).pth", map_location=torch.device("cpu")))  # .40
net.append(model)
