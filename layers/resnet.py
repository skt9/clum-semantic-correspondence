import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm



def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
    return str(device), enabled, out_dtype

class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False,
                      early_exit = False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        print(f"x.device: {x.device}")
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        # with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            net = self.net
            feats = {1:x}       #   feature pyramid
            im_shape = x.shape[2:]
            print(f"im_shape: {im_shape}")
            print(f"feats[1].shape: {feats[1].shape}")
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x 
            print(f"feats[2].shape: {feats[2].shape}")
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x 
            # print(f"feats[4].shape: {feats[4].shape}")
            feats[4] = F.interpolate(feats[4],im_shape,mode="bicubic",align_corners=True)
            print(f"feats[4].shape: {feats[4].shape}")
            x = net.layer2(x)
            feats[8] = x
            feats[8] = F.interpolate(feats[8],im_shape,mode="bicubic",align_corners=True)
            print(f"feats[8].shape: {feats[8].shape}")
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            feats[16] = F.interpolate(feats[16],im_shape,mode="bicubic",align_corners=True)
            print(f"feats_16.shape: {feats[16].shape}")
            # print(f"feats[16].shape: {feats[16].shape}")
            x = net.layer4(x)
            feats[32] = x
            # print(f"feats[32].shape: {feats[32].shape}")
            feats[32] = F.interpolate(feats[32],im_shape,mode="bicubic",align_corners=True)
            # print(f"feats_32.shape: {feats_32.shape}")
            feats_combined = torch.cat([feats[32],feats[16],feats[8]],dim=1)
            return feats_combined

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass
