import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pk

class ResampleLoss(nn.Module):

    def __init__(self, device,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm
        self.device      = device

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha'] # change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = torch.from_numpy(np.asarray(class_freq)).float().to(self.device)
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8

        self.freq_inv = torch.ones(self.class_freq.shape).to(self.device) / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(-logpt)
            wtloss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            alpha_t = torch.where(label==1, self.alpha, 1-self.alpha)
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss ####################### balance_param should be a tensor
            loss = reduce_loss(loss, reduction)             ############################ add reduction
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None): 
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).to(self.device) / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).to(self.device)
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).to(self.device) / \
                     (1 - torch.pow(self.CB_beta, avg_n)).to(self.device)
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).to(self.device) / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).to(self.device)
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).to(self.device) / \
                     (1 - torch.pow(self.CB_beta, min_n)).to(self.device)
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight
    

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss



import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# gamma : 0.25, 0.5, 1.0, 2.0
# alpha 0.25, 0.5, 1.0, 2.0


class FocalLoss11(nn.Module):
    
    def __init__(self, device, pos_weight, alpha=.25, gamma=2, reduction='none'):
        super(FocalLoss11, self).__init__()
        self.gamma  = gamma
        self.pos_weight = pos_weight
        self.device     = device
        self.reduction  = reduction
        
    def forward(self, y_pred, y_true):
        # y_pred is the logits without Sigmoid
        assert y_pred.shape == y_true.shape
        pt = torch.exp(-F.binary_cross_entropy_with_logits(y_pred, y_true, reduction=self.reduction)).detach()
        sample_weight = (1 - pt) ** self.gamma
        sample_weight = sample_weight.to(self.device)
        return F.binary_cross_entropy_with_logits(y_pred, y_true, weight=sample_weight, pos_weight=self.pos_weight.to(self.device))
    

class FocalLoss22(nn.Module):
    
    def __init__(self, device, alpha=.25, gamma=2, reduction='none'):
        super(FocalLoss22, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype = torch.float32).to(device)
        self.reduction  = reduction

        
    def forward(self, input, target):
        
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction).to(self.device)
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss =  (1-pt)**self.gamma * BCE_loss
        
        return F_loss.mean()
    
    

class FocalLoss33(nn.Module):
    
    def __init__(self, device, alpha=.25, gamma=2, reduction='none'):
        super(FocalLoss33, self).__init__()
        self.gamma  = gamma
        self.reduction  = reduction
    
    def forward(self, logits, targets):
        
        num_label = targets.shape[1]
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma)
        
        if self.reduction=='none':
            loss = loss
        elif self.reduction=='mean':
            loss = num_label*loss.mean()
        elif self.reduction=='none':
            loss = num_label*loss.sum()
            
        return loss

#no
class FocalLoss44(nn.Module):
    def __init__(self, device, alpha=0.25, gamma=2, reduction='none'):
        super(FocalLoss44, self).__init__()
        self._gamma = gamma
        self._alpha = alpha
        self.reduction  = reduction

    def forward(self, y_pred, y_true):
        
        
        cross_entropy_loss = torch.nn.BCELoss()(y_pred,y_true)
        p_t = ((y_true * y_pred) +
               ((1 - y_true) * (1 - y_pred)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha +
                                   (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)
        
        if self.reduction == 'none':
            return focal_cross_entropy_loss
        elif self.reduction == 'mean':
            return focal_cross_entropy_loss.mean()
        elif self.reduction == 'sum':
            return focal_cross_entropy_loss.sum()
            
    
    



class DICE_LOSS11(nn.Module):
    def __init__(self, device, alpha=0.25, gamma=2, reduction='none'):
        super(DICE_LOSS11, self).__init__()
        self._gamma = gamma
        self._alpha = alpha
        self.reduction  = reduction

    def forward(self, y_pred, y_true):
        
        y_pred = torch.sigmoid(y_pred)
        smooth = 1.

        iflat = y_pred.view(-1)
        tflat = y_true.view(-1)
        if self.reduction=='none':
            intersection = (iflat * tflat)
        elif self.reduction=='mean':
            intersection = (iflat * tflat).mean
        elif self.reduction=='sum':
            intersection = (iflat * tflat).sum()
            
        

        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))
    

    
class FocalLoss55(nn.Module):
    def __init__(self, device, alpha=0.25, gamma=2, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.reduction  = reduction

        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        if self.reduction=='none':
            return loss
        elif self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
            
        


class FocalLoss66(nn.Module):
    def __init__(self, device, alpha=0.25, gamma=2, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.reduction  = reduction
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = torch.log((1-target) + torch.sigmoid(input) * (target * 2 -1))

        loss = (invprobs * self.gamma).exp() * loss
        
        if self.reduction=='none':
            return loss
        elif self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
    
    
    
import torch
from torch import nn
import torch.nn.functional as F
# from config import DefaultConfig

    


class BinaryFocalLoss1111(nn.Module):

    def __init__(self, DEVICE, alpha=0.25, gamma=2, size_average=True):
        super(BinaryFocalLoss1111, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.size_average = size_average
        self.DEVICE = DEVICE

    def forward(self, pred, target):

        # pred = nn.Sigmoid()(pred)
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)


        pred = torch.cat((1-pred, pred), dim=1).to(self.DEVICE)


        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).to(self.DEVICE)
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        log_p = probs.log()

 
        alpha = torch.ones(pred.shape[0], pred.shape[1]).to(self.DEVICE)
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

 
        batch_loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p

 
        if self.size_average == 'mean':
            loss = batch_loss.mean()
            return loss
        elif self.size_average == 'sum':
            loss = batch_loss.sum()
            return loss
        else:
            return loss


class FocalLossMultiLabel1111(nn.Module):
 
    def __init__(self, device , alpha=0.25, gamma=2, reduction='none'):
        super(FocalLossMultiLabel1111, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        self.DEVICE = device
        if reduction=='mean':
            self.size_average = 'mean'
            
        elif reduction=='sum':
            self.size_average = 'sum'
            
        elif reduction=='none':
            self.size_average = 'none'
            

    def forward(self, pred, target):
        criterion = BinaryFocalLoss1111(self.DEVICE, self.alpha, self.gamma, self.size_average)
        loss = torch.zeros(1, target.shape[1]).to(self.DEVICE)

 
        for label in range(target.shape[1]):
            batch_loss = criterion(pred[:, label].to(self.DEVICE), target[:, label])
            loss[0, label] = batch_loss.mean()

 
        if self.size_average:
            loss = loss.mean().to(self.DEVICE)
        else:
            loss = loss.sum().to(self.DEVICE)

        return loss
    


    
    
class BinaryDiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self, device, alpha=0.5, gamma=0.5, reduction='none'):
        super(BinaryDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction= reduction

    def forward(self, y_pred, y_true):
        """
        :param y_pred: [N, C, ]
        :param y_true: [N, C, ]
        :param reduction: 'mean' or 'sum'
        """
        batch_size = y_true.size(0)
        y_pred = y_pred.contiguous().view(batch_size, -1)
        y_true = y_true.contiguous().view(batch_size, -1)

        numerator = torch.sum(2 * torch.pow((1 - y_pred), self.alpha) * y_pred * y_true, dim=1) + self.gamma
        denominator = torch.sum(torch.pow((1 - y_pred), self.alpha)  * y_pred + y_true, dim=1) + self.gamma
        loss = 1 - (numerator / denominator)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

        
        
class DiceLoss(nn.Module):
    def __init__(self, device, alpha=1, gamma=1,reduction='none'):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.binary_dice_loss = BinaryDiceLoss(alpha, gamma)
        self.reduction= reduction

    def forward(self, y_pred, y_true):
        """
        :param y_pred: [N, C, ]
        :param y_true: [N, ]
        :param reduction: 'mean' or 'sum'
        """
        shape = y_pred.shape
        num_labels = shape[1]
        dims = [i for i in range(len(y_pred.shape))]
        dims.insert(1, dims.pop())
        y_pred = torch.softmax(y_pred, dim=1)
        loss = self.binary_dice_loss(y_pred, y_true, reduction)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss    
    



class DiceLoss12(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)
    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
        >>> input = torch.FloatTensor([2, 1, 2, 2, 1])
        >>> input.requires_grad=True
        >>> target = torch.LongTensor([0, 1, 0, 0, 0])
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self,
                 device,
                 smooth: Optional[float] = 1e-4,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratio: float = 0.0,
                 alpha: float = 0.0,
                 reduction: Optional[str] = "mean",
                 index_label_position=True) -> None:
        super(DiceLoss12, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits_size = input.shape[-1]

        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            loss = 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = target.float()
        flat_input = torch.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0 :
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = torch.masked_select(flat_input, neg_example.view(-1, 1).bool()).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = (torch.argmax(flat_input, dim=1) == label_idx & flat_input[:, label_idx] >= threshold) | pos_example.view(-1)
                    ohem_mask_idx = torch.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num+1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


##
# version 1: use torch.autograd
class FocalLossV1(nn.Module):

    def __init__(self,device,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
    




##
# version 2: user derived grad computation
class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        #  logits = logits.float()

        probs = torch.sigmoid(logits)
        coeff = (label - probs).abs_().pow_(gamma).neg_()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        ce_term1 = log_probs.mul_(label).mul_(alpha)
        ce_term2 = log_1_probs.mul_(1. - label).mul_(1. - alpha)
        ce = ce_term1.add_(ce_term2)
        loss = ce * coeff

        ctx.vars = (coeff, probs, ce, label, gamma, alpha)

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, ce, label, gamma, alpha) = ctx.vars

        d_coeff = (label - probs).abs_().pow_(gamma - 1.).mul_(gamma)
        d_coeff.mul_(probs).mul_(1. - probs)
        d_coeff = torch.where(label < probs, d_coeff.neg(), d_coeff)
        term1 = d_coeff.mul_(ce)

        d_ce = label * alpha
        d_ce.sub_(probs.mul_((label * alpha).mul_(2).add_(1).sub_(label).sub_(alpha)))
        term2 = d_ce.mul(coeff)

        grads = term1.add_(term2)
        grads.mul_(grad_output)

        return grads, None, None, None


class FocalLossV2(nn.Module):

    def __init__(self,device,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV2()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss



# logits, target
class BPMLLLoss(torch.nn.Module):
    def __init__(self, device, bias=(1, 1), reduction='none'):
        super(BPMLLLoss, self).__init__()
        self.bias = bias

    def forward(self, c: Tensor, y: Tensor) -> Tensor:
        r"""
        compute the loss, which has the form:
        L = \sum_{i=1}^{m} \frac{1}{|Y_i| \cdot |\bar{Y}_i|} \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        :param c: prediction tensor, size: batch_size * n_labels
        :param y: target tensor, size: batch_size * n_labels
        :return: size: scalar tensor
        """
        y = y.float()
        y_bar = -y + 1
        y_norm = torch.pow(y.sum(dim=(1,)), self.bias[0])
        y_bar_norm = torch.pow(y_bar.sum(dim=(1,)), self.bias[1])
        assert torch.all(y_norm != 0) or torch.all(y_bar_norm != 0), "an instance cannot have none or all the labels"
        return torch.mean(1 / torch.mul(y_norm, y_bar_norm) * self.pairwise_sub_exp(y, y_bar, c))

    def pairwise_sub_exp(self, y: Tensor, y_bar: Tensor, c: Tensor) -> Tensor:
        r"""
        compute \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        """
        truth_matrix = y.unsqueeze(2).float() @ y_bar.unsqueeze(1).float()
        exp_matrix = torch.exp(c.unsqueeze(1) - c.unsqueeze(2))
        if reduction=='sum':
            return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))
        elif reduction=='mean':
            return (torch.mul(truth_matrix, exp_matrix)).mean(dim=(1, 2))
        else:
            return torch.mul(truth_matrix, exp_matrix)



    


##
# v2: self-derived grad formula
class SoftDiceLossV2(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,device,
                 p=1,
                 smooth=1, reduction='none'):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV2Func.apply(logits, labels, self.p, self.smooth)
        return loss


class SoftDiceLossV2Func(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        '''
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        '''
        #  logits = logits.float()

        probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=1) + smooth
        denor = (probs.pow(p) + labels.pow(p)).sum(dim=1) + smooth
        loss = 1. - numer / denor

        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of soft-dice loss
        '''
        probs, labels, numer, denor, p, smooth = ctx.vars

        numer, denor = numer.view(-1, 1), denor.view(-1, 1)

        term1 = (1. - probs).mul_(2).mul_(labels).mul_(probs).div_(denor)

        term2 = probs.pow(p).mul_(1. - probs).mul_(numer).mul_(p).div_(denor.pow_(2))

        grads = term2.sub_(term1).mul_(grad_output)

        return grads, None, None, None
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp




## Soft Dice Loss for binary segmentation
##
# v1: pytorch autograd
class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,device,
                 p=1,
                 smooth=1, reduction='none'):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss

    
    
class Dual_Focal_loss(nn.Module):
    '''
    This loss is proposed in this paper: https://arxiv.org/abs/1909.11932
    It does not work in my projects, hope it will work well in your projects.
    Hope you can correct me if there are any mistakes in the implementation.
    '''

    def __init__(self, device, ignore_lb=255, eps=1e-5, reduction='mean'):
        super(Dual_Focal_loss, self).__init__()
        self.ignore_lb = ignore_lb
        self.eps = eps
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')
        self.device = device

    def forward(self, logits, label):
        ignore = label.data.to(self.device) == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits

        pred = torch.softmax(logits, dim=1)
        loss = -torch.log(self.eps + 1. - self.mse(pred, lb_one_hot)).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss
    

    

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, device, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
                 disable_torch_grad_focal_loss=True, reduction='none'):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction= reduction

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        if self.reduction=='mean':
            return -loss.mean()
        elif self.reduction=='sum':
            return -loss.sum()
        else:
            return -loss
            
        

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, device, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
                 disable_torch_grad_focal_loss=False, reduction='none'):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction= reduction


        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
        
        if self.reduction=='mean':
            return -self.loss.mean()
        elif self.reduction=='sum':
            return -self.loss.sum()
        else:
            return -self.loss

# for 50

gamma_values      = {0: 0.16, 1: 0.15}

# for full data
gamma_values_full = {0: 0.05, 1: 0.04}



with open('./data/loss_data/class_freq', 'rb') as f:
    all_data = pk.load(f)
    
with open('./data/loss_data/pos_weights', 'rb') as f:
    all_data_pos = pk.load(f)
    

    
half_50  = all_data['half_50']
full_50  = all_data['full_50']
full_all = all_data['full_all']
half_all = all_data['half_all']

half_50_pos  = all_data_pos['half_50_pos']
full_50_pos  = all_data_pos['full_50_pos']
full_all_pos = all_data_pos['full_all_pos']
half_all_pos = all_data_pos['half_all_pos']

half_ = {0: half_50, 1: full_50}
full_ = {0: half_all,1: full_all}

half_pos =  {0: half_50_pos,  1: full_50_pos}
full_pos =  {0: half_all_pos, 1: full_all_pos}


def get_l(loss_func_name, train_num, label_level, problem_name, device, args):
    
    if problem_name == 'mimic-iii_2_50':
        cluster_    = half_
        pos_wei     = half_pos
    else:
        cluster_    = full_
        pos_wei     = full_pos
    
    class_freq        = cluster_[label_level]
    pos_weight        = pos_wei[label_level]
    
    
    if loss_func_name == 'BCE':
        loss_func = ResampleLoss(device,reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=False, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'FL':
        loss_func = ResampleLoss(device,reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'CBloss': #CB
        loss_func = ResampleLoss(device,reweight_func='CB', loss_weight=10.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                 class_freq=class_freq, train_num=train_num) 

    if loss_func_name == 'R-BCE-Focal': # R-FL
        loss_func = ResampleLoss(device,reweight_func='rebalance', loss_weight=1.0, 
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 map_param=dict(alpha=0.1, beta=10.0, gamma=gamma_values[label_level]), 
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'NTR-Focal': # NTR-FL
        loss_func = ResampleLoss(device,reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'DBloss-noFocal': # DB-0FL
        loss_func = ResampleLoss(device,reweight_func='rebalance', loss_weight=0.5,
                                 focal=dict(focal=False, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 map_param=dict(alpha=0.1, beta=10.0, gamma=gamma_values[label_level]), 
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'CBloss-ntr': # CB-NTR
        loss_func = ResampleLoss(device,reweight_func='CB', loss_weight=10.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'DBloss': # DB
        loss_func = ResampleLoss(device,reweight_func='rebalance', loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 map_param=dict(alpha=0.1, beta=10.0, gamma=gamma_values[label_level]), 
                                 class_freq=class_freq, train_num=train_num)
    
    
    if loss_func_name == 'base':
        loss_func = nn.BCEWithLogitsLoss()
        
    
    if loss_func_name == 'FocalLoss11':
        loss_func = FocalLoss11(device = device, pos_weight = pos_weight, reduction=args.reduction)
    
    if loss_func_name == 'FocalLoss22':
        loss_func = FocalLoss22(device = device, reduction=args.reduction)
        
    if loss_func_name == 'FocalLoss44':
        loss_func = FocalLoss44(device = device, reduction=args.reduction)
        
    if loss_func_name == 'FocalLoss33':
        loss_func = FocalLoss33(device = device, reduction=args.reduction)
        
    if loss_func_name == 'DICE_LOSS11':
        loss_func = DICE_LOSS11(device = device, reduction=args.reduction)
    
    if loss_func_name == 'FocalLoss55':
        loss_func = FocalLoss55(device = device, reduction=args.reduction)
        
    if loss_func_name == 'FocalLoss66':
        loss_func = FocalLoss66(device = device, reduction=args.reduction)
        
    if loss_func_name == 'FocalLossMultiLabel1111':
        loss_func = FocalLossMultiLabel1111(device = device, reduction=args.reduction)
        
    if loss_func_name == 'DiceLoss':
        loss_func = DiceLoss(device = device, reduction=args.reduction)
        
    if loss_func_name == 'BinaryDiceLoss':
        loss_func = BinaryDiceLoss(device = device, reduction=args.reduction)
        
    if loss_func_name == 'DiceLoss12':
        loss_func = DiceLoss12(device = device, reduction=args.reduction)
        
    if loss_func_name == 'FocalLossV1':
        loss_func = FocalLossV1(device = device, reduction=args.reduction)
        
    if loss_func_name == 'FocalLossV2':
        loss_func = FocalLossV2(device = device, reduction=args.reduction)
        
    if loss_func_name == 'BPMLLLoss':
        loss_func = BPMLLLoss(device = device, reduction=args.reduction)
        
    if loss_func_name == 'SoftDiceLossV2':
        loss_func = SoftDiceLossV2(device = device, reduction=args.reduction)
        
    if loss_func_name == 'SoftDiceLossV1':
        loss_func = SoftDiceLossV1(device = device, reduction=args.reduction)
        
    if loss_func_name == 'Dual_Focal_loss':
        loss_func = Dual_Focal_loss(device = device, reduction=args.reduction)
        
    if loss_func_name == 'AsymmetricLoss':
        loss_func = AsymmetricLoss(device = device, reduction=args.reduction)
        
    if loss_func_name == 'AsymmetricLossOptimized':
        loss_func = AsymmetricLossOptimized(device = device, reduction=args.reduction)
    
    
       
    return loss_func
