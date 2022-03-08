import numpy as np 
import torch
import torch.nn.functional as F
from utils.cfg import config as cfg

class metric():
    def __init__(self, pred, label):
    ### Input 
    ###   pred, label : 1 X W X H torch tensor
    ### Output
    ###   1 torch tensor
        pred_f = torch.flatten(torch.sigmoid(pred))
        label_f = torch.flatten(label)
        intersection = torch.sum(pred_f * label_f)
        self.DSC = 2. * (intersection + cfg.smooth) / (torch.sum(pred_f) + torch.sum(label_f) + cfg.smooth)

    def cal_DSC(self):
        return self.DSC

    def DSC_loss(self):
        return 1 - self.DSC

    def Jaccard(self):
        return self.DSC / (2. - self.DSC)

    def Jaccard_loss(self):
        return 1 - self.Jaccard()

    def total_loss(self):
        return self.DSC_loss() + self.Jaccard_loss()