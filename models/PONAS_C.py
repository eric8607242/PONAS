import math

import torch
import torch.nn as nn

def conv_1x1_bn(input_depth, output_depth):
    return nn.Sequential(
        nn.Conv2d(input_depth, output_depth, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_depth),
        nn.ReLU6(inplace=True)
    )

class ConvBNRelu(nn.Sequential):
    def __init__(self, 
                 input_depth,
                 output_depth,
                 kernel,
                 stride,
                 pad,
                 activation="relu",
                 group=1,
                 *args,
                 **kwargs):

        super(ConvBNRelu, self).__init__()

        assert activation in ["hswish", "relu", None]
        assert stride in [1, 2, 4]

        self.add_module("conv", nn.Conv2d(input_depth, output_depth, kernel, stride, pad, groups=group, bias=False))
        self.add_module("bn", nn.BatchNorm2d(output_depth))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())

class SEModule(nn.Module):
    reduction = 4
    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = nn.Conv2d(C, mid, 1, 1, 0)
        conv2 = nn.Conv2d(mid, C, 1, 1, 0)

        self.operation = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()    
            )

    def forward(self, x):
        return x * self.operation(x)

class MBConv(nn.Module):
    def __init__(self,
                 input_depth,
                 output_depth,
                 expansion,
                 kernel,
                 stride,
                 activation,
                 group=1,
                 se=False,
                 *args,
                 **kwargs):
        super(MBConv, self).__init__()
        self.use_res_connect = True if (stride==1 and input_depth == output_depth) else False
        mid_depth = int(input_depth * expansion)

        self.group = group

        if input_depth == mid_depth:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNRelu(input_depth,
                                         mid_depth,
                                         kernel=1,
                                         stride=1,
                                         pad=0,
                                         activation=activation,
                                         group=group,
                                    )

        self.depthwise = ConvBNRelu(mid_depth,
                                    mid_depth,
                                    kernel=kernel,
                                    stride=stride,
                                    pad=(kernel//2),
                                    activation=activation,
                                    group=mid_depth,
                                )

        self.point_wise_1 = ConvBNRelu(mid_depth,
                                       output_depth,
                                       kernel=1,
                                       stride=1,
                                       pad=0,
                                       activation=None,
                                       group=group,
                                    )
        self.se = SEModule(mid_depth) if se else None

    def forward(self, x):
        y = self.point_wise(x)
        y = self.depthwise(y)

        y = self.se(y) if self.se is not None else y
        y = self.point_wise_1(y)

        y = y + x if self.use_res_connect else y
        return y



class POBlock(nn.Module):
    def __init__(self, 
                 input_depth,
                 output_depth,
                 kernel,
                 stride,
                 activation="relu",
                 block_type="MB",
                 expansion=1,
                 group=1,
                 se=False):

        super(POBlock, self).__init__()

        self.block = MBConv(input_depth,
                            output_depth,
                            expansion,
                            kernel,
                            stride,
                            activation,
                            group,
                            se)         


    def forward(self, x):
        y = self.block(x)
        return y



class PONASC(nn.Module):
    def __init__(self, classes=1000):
        super(PONASC, self).__init__()


        model_cfg = [
                [6, 5, False, 2, 32], 
                [3, 7, True, 1, 32], 
                [6, 5, True, 2, 40], 
                [3, 5, True, 1, 40], 
                [3, 5, False, 1, 40], 
                [3, 7, False, 1, 40], 
                [6, 7, False, 2, 80], 
                [3, 5, True, 1, 80], 
                [3, 7, True, 1, 80], 
                [6, 5, True, 1, 80], 
                [6, 5, True, 1, 96], 
                [3, 7, False, 1, 96], 
                [3, 7, False, 1, 96], 
                [6, 5, True, 1, 96], 
                [6, 7, True, 2, 192], 
                [3, 7, True, 1, 192], 
                [3, 5, False, 1, 192], 
                [3, 7, True, 1, 192], 
                [6, 5, True, 1, 320]]



        self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=2, pad=3//2, activation="relu")

        self.stages = nn.Sequential()
        self.stages.add_module("0", POBlock(32, 16, 3, 1))
        input_depth = 16
        for i, cfg in enumerate(model_cfg):
            e, k, se, s, output_depth = cfg

            self.stages.add_module(str(i+1), POBlock(input_depth=input_depth, 
                                                           output_depth=output_depth,
                                                           kernel=k,
                                                           stride=s,
                                                           expansion=e,
                                                           se=se))
            input_depth = output_depth


        self.stages.add_module("last", conv_1x1_bn(input_depth, 1280))
        self.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1280, classes)
                )
        
        self._initialize_weights()

    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        y = y.mean(3).mean(2)
        y = self.classifier(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
