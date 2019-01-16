from torch import nn
from loss.loss_utils import *



class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = ae_loss
        self.regr_loss   = regr_loss

    def forward(self, outs, targets):
        stride = 6

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tl_tags  = outs[2::stride]
        br_tags  = outs[3::stride]
        tl_regrs = outs[4::stride]
        br_regrs = outs[5::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]


        # focal loss
        focal_loss = 0

        tl_heats = [sigmoid(t) for t in tl_heats]
        br_heats = [sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_losses = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_losses += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_losses += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_losses = self.regr_weight * regr_losses
        loss = (focal_loss + pull_loss + push_loss + regr_losses) / len(tl_heats)
        return loss.unsqueeze(0)
