"""
"""

import torch
from torch import nn

class TargetWeightedMSELoss(nn.Module):
    """
    Target weigted MSE loss.
    """
    def __init__(self, weight_pow):
        """
        Input:
            - weight_pow: a positive number
        """
        super().__init__()
        assert weight_pow > 0, \
            'weight_pow must be a postive number!'
        self.weight_pow = weight_pow

    def forward(self, data, target):
        """
        Input:
            - input_x: the approximation to the the target tensor.
            - target: the ground truth tensor.
        """
        weight = torch.pow(torch.abs(target), self.weight_pow)
        diff = (data - target) ** 2 * weight
        return diff.sum() / weight.sum()


class TargetWeightedL1Loss(nn.Module):
    """
    Target weigted MSE loss.
    """
    def __init__(self, weight_pow):
        """
        Input:
            - weight_pow: a positive number
        """
        super().__init__()
        assert weight_pow > 0, \
            'weight_pow must be a postive number!'
        self.weight_pow = weight_pow

    def forward(self, data, target):
        """
        Input:
            - input_x: the approximation to the the target tensor.
            - target: the ground truth tensor.
        """
        weight = torch.pow(torch.abs(target), self.weight_pow)
        diff = np.abs(data - target) * weight
        return diff.sum() / weight.sum()


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced classification.
    From "Focal loss for dense object detection"
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, gamma, eps=1e-8):
        """
        Input:
            - gamma:
            - eps:
        """
        super().__init__()
        self.gamma, self.eps = gamma, eps

    def forward(self, pred, label):
        """
        Input:
        Output:
        """
        p_0, p_1 = 1 - pred + self.eps, pred + self.eps
        l_0, l_1 = ~label, label
        focal_loss = (
            torch.pow(p_1, self.gamma) * torch.log2(p_0) * l_0 +
            torch.pow(p_0, self.gamma) * torch.log2(p_1) * l_1
        )
        return -torch.mean(focal_loss)


class BCAELoss(nn.Module):
    """
    BCAE loss and metrics class
    """

    # pylint: disable=too-many-arguments
    def __init__(self, *,
                 clf_threshold  = .5,
                 gamma          = 2,
                 eps            = 1e-8,
                 reg_loss       = 'mse',
                 weight_pow     = None):
        """
        Initialize the parameters
        """
        super().__init__()

        self.clf_threshold = clf_threshold

        # Classification loss
        if gamma is None:
            self.clf_loss_fn = nn.BCELoss()
        else:
            self.clf_loss_fn = FocalLoss(gamma, eps)

        # Regression loss
        if weight_pow is None:
            if reg_loss == 'mse':
                self.reg_loss_fn = nn.MSELoss()
            else:
                self.reg_loss_fn = nn.L1Loss()
        else:
            if reg_loss == 'mse':
                self.reg_loss_fn = TargetWeightedMSELoss(weight_pow)
            else:
                self.reg_loss_fn = TargetWeightedL1Loss(weight_pow)

        # bcae training uses dynamic coefficient
        # in the linear combination of ckassification
        # and regression loss
        # Specifically, the coefficient of classification loss
        # will be scaled up to match that of regression loss
        # Find the formula in the pipe function below.
        self.clf_coef = 2000
        self.expo = .5


    def forward(self, clf_output, reg_output, tag, adc):
        """
        Input:
        Output:
        """

        mask = clf_output > self.clf_threshold
        combined = reg_output * mask

        with torch.no_grad():
            pos = mask.sum()
            true = tag.sum()
            true_pos = (mask * tag).sum()
            mse = torch.pow(combined - adc, 2).mean()

        # Classification and regression loss
        loss_clf = self.clf_loss_fn(clf_output, tag)
        loss_reg = self.reg_loss_fn(combined, adc)

        # update the coefficient for classification loss
        if torch.isnan(loss_clf):
            self.clf_coef = 2000
        else:
            self.clf_coef = ( self.expo * self.clf_coef
                              + (loss_reg / (loss_clf)).item() ) / (self.expo + 1.)
        # self.clf_coef = min(self.clf_coef, 10)

        # save all type of losses to a dictionary
        losses = {'loss': loss_reg + self.clf_coef * loss_clf,
                  'clf loss': loss_clf.item(),
                  'reg loss': loss_reg.item(),
                  'clf coef': self.clf_coef,
                  'pos': pos.item(),
                  'true': true.item(),
                  'true pos': true_pos.item(),
                  'mse': mse.item()}

        return losses


class BCAELossSoft(nn.Module):
    """
    BCAE loss and metrics class
    """

    # pylint: disable=too-many-arguments
    def __init__(self, *,
                 clf_threshold  = .5,
                 gamma          = 2,
                 eps            = 1e-8,
                 reg_loss       = 'mse',
                 weight_pow     = None,
                 alpha          = 4.):
        """
        Initialize the parameters
        """
        super().__init__()

        self.clf_threshold = clf_threshold

        # Classification loss
        if gamma is None:
            self.clf_loss_fn = nn.BCELoss()
        else:
            self.clf_loss_fn = FocalLoss(gamma, eps)

        # Regression loss
        if weight_pow is None:
            if reg_loss == 'mse':
                self.reg_loss_fn = nn.MSELoss()
            else:
                self.reg_loss_fn = nn.L1Loss()
        else:
            if reg_loss == 'mse':
                self.reg_loss_fn = TargetWeightedMSELoss(weight_pow)
            else:
                self.reg_loss_fn = TargetWeightedL1Loss(weight_pow)

        self.eps = eps
        self.alpha = alpha

        # bcae training uses dynamic coefficient
        # in the linear combination of ckassification
        # and regression loss
        # Specifically, the coefficient of classification loss
        # will be scaled up to match that of regression loss
        # Find the formula in the pipe function below.
        self.clf_coef = 2000
        self.expo = .5


    def forward(self, clf_output, reg_output, tag, adc):
        """
        Input:
        Output:
        """

        # mask, no longer differentiable
        mask = clf_output > self.clf_threshold

        # soft mask. Let the two decoder colaborate
        beta_zero = self.clf_threshold / (1. - self.clf_threshold)
        beta = (clf_output + self.eps) / (1. - clf_output + self.eps)
        soft_mask = 1. / (1. + torch.pow(beta_zero / beta, self.alpha))

        combined = reg_output * soft_mask

        with torch.no_grad():
            pos = mask.sum()
            true = tag.sum()
            true_pos = (mask * tag).sum()
            mse = torch.pow(combined - adc, 2).mean()

        # Classification and regression loss
        loss_clf = self.clf_loss_fn(clf_output, tag)
        loss_reg = self.reg_loss_fn(combined, adc)

        # update the coefficient for classification loss
        if torch.isnan(loss_clf):
            self.clf_coef = 2000
        else:
            self.clf_coef = ( self.expo * self.clf_coef
                              + (loss_reg / (loss_clf)).item() ) / (self.expo + 1.)
        # self.clf_coef = min(self.clf_coef, 10)

        # save all type of losses to a dictionary
        losses = {'loss': loss_reg + self.clf_coef * loss_clf,
                  'clf loss': loss_clf.item(),
                  'reg loss': loss_reg.item(),
                  'clf coef': self.clf_coef,
                  'pos': pos.item(),
                  'true': true.item(),
                  'true pos': true_pos.item(),
                  'mse': mse.item()}

        return losses
