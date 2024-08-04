import torch
import torch.nn as nn
import torch.nn.functional as F

class HammingLoss(torch.nn.Module):
    def forward(self, unaries_predicted:torch.Tensor, unaries_target:torch.Tensor):
        unaries_errors = unaries_predicted * (1.0 - unaries_target) + (1.0 - unaries_predicted) * unaries_target
        return unaries_errors.sum()

class CycleLoss(nn.Module):

    def __init__(self, loss_type = "L1"):
        super(CycleLoss, self).__init__()
        if loss_type == "L1":
            loss = nn.L1Loss()
            self.loss = loss
        elif loss_type == "L2":
            loss = nn.L2Loss()
            self.loss = loss

    def forward(self, x12: torch.Tensor, x23: torch.Tensor, x31: torch.Tensor):
        assert(x12.shape==x23.shape)
        assert(x23.shape==x31.shape)
        cycle = torch.matmul(torch.matmul(x12,x23),x31)
        eye = torch.eye(x12.shape[0]).to(cycle.device)
        return self.loss(cycle, eye)
