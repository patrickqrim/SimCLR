import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function, Variable
import torch.optim as optim
from torch.optim import lr_scheduler

from exceptions.exceptions import InvalidBackboneError

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class WSCNetSimCLR(nn.Module):

    def __init__(self, out_dim):
        super(WSCNetSimCLR, self).__init__()

        self.model_name = "resnet"
        self.out_dim = out_dim
        self.num_maps = 4
        self.feature_extract = True
        self.use_pretrained = True

        self.backbone, input_size = initialize_model(self.model_name,
                                                    self.out_dim,
                                                    self.num_maps,
                                                    self.feature_extract,
                                                    self.use_pretrained)

        # add mlp projection head
        self.backbone.classifier = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), self.backbone.classifier)
    
    def get_optimizer(self):
        params_to_update = self.backbone.parameters()
        if self.feature_extract:
            params_to_update = []
            for name,param in self.backbone.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        else:
            for name,param in self.backbone.named_parameters():
                if param.requires_grad == True:
                    continue

        optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)
        scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        return optimizer_ft, scheduler_ft

    def forward(self, x):
        return self.backbone(x)




#------------------------------------WSCNet------------------------------------#

class ResNetWSL(nn.Module):
    
    def __init__(self, model, num_classes, num_maps, pooling, pooling2):
        super(ResNetWSL, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        # print(str(list(model.children())) + '\n\n\n')
        self.num_ftrs = model.fc.in_features  # =2048

        self.downconv = nn.Sequential(
            nn.Conv2d(self.num_ftrs, num_classes*num_maps, kernel_size=1,
             stride=1, padding=0, bias=True))
        
        self.GAP = nn.AvgPool2d(14)
        self.GMP = nn.MaxPool2d(14) # unused
        self.spatial_pooling = pooling
        self.spatial_pooling2 = pooling2
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x_ori = x  
        
        # Create v
        x = self.downconv(x) 
        x_conv = x # the output of 1x1 convolutions
        x = self.GAP(x) # height and depth-wise to get kC x 1 vector
        x = self.spatial_pooling(x) # depth-wise to get v
        x = x.view(x.size(0), -1) # get rid of excess dimensions? (still v)

        # Applying the sentiment map
        x_conv = self.spatial_pooling(x_conv) # depth-wise (Figure 4, average pooling)
        x_conv = x_conv * x.view(x.size(0),x.size(1),1,1) # element-wise multiplication of v and x_conv (Figure 4)
        x_conv = self.spatial_pooling2(x_conv) # sum (or average?) x_conv to get SENTIMENT MAP
        x_conv_copy = x_conv # make a copy of SENTIMENT MAP
        for num in range(0,2047):            
            x_conv_copy = torch.cat((x_conv_copy, x_conv),1) # 2048 copies of the sentiment map to do Hadamard product
        x_conv_copy = torch.mul(x_conv_copy,x_ori) # Hadamard product
        x_conv_copy = torch.cat((x_ori,x_conv_copy),1) # concatenating original to Hadamard product
        x_conv_copy = self.GAP(x_conv_copy) # height and depth-wise to get 2*2048 x 1 x 1
        x_conv_copy = x_conv_copy.view(x_conv_copy.size(0),-1) # flatten
        x_conv_copy = self.classifier(x_conv_copy) # return logits
        return x_conv_copy



class ClassWisePoolFunction(Function):
    #def __init__(self, num_maps):
        #super(ClassWisePoolFunction, self).__init__()
        
    @staticmethod
    def forward(self, input, num_maps):
        self.num_maps = num_maps
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w), None

class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction.apply(input, self.num_maps)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, num_maps, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 224
    
    if model_name == "resnet":
        """ Resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        # num_maps is WHAT THEY DIVIDE BY!! so if input = kC, num_maps = k, then output = C 
        pooling = nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling2 = nn.Sequential()
        pooling2.add_module('class_wise', ClassWisePool(num_classes))
        model_ft = ResNetWSL(model_ft, num_classes, num_maps, pooling, pooling2)
        
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size