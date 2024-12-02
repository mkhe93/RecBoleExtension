# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def forwardforward_loss_fn(y, theta, target):  # target=1 if positive, 0 otherwise.
    if isinstance(target, (int, float)):  # if sign (1 or -1) is given, turn it into a target (1 or 0)
        target = max(0.0, float(target))

    if isinstance(target, (int, float)):
        target = torch.tensor([target] * len(y), device=y.device, dtype=torch.float)

    logits = y.pow(2).sum(dim=1) - theta
    with torch.no_grad():
        accumulated_logits = logits.mean().item()

    loss = F.binary_cross_entropy_with_logits(input=logits, target=target, reduction='mean')
    return loss, accumulated_logits


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class ALSLoss(nn.Module):
    """Alternating Least Squares (ALS) Loss function."""

    def __init__(self, reg_weight=0.01):
        super(ALSLoss, self).__init__()
        self.reg_weight = reg_weight  # Regularization term

    def forward(self, user_factors, item_factors, interaction_matrix, confidence_matrix):
        """
        Compute the ALS loss.

        Args:
            user_factors (torch.Tensor): User latent factors matrix.
            item_factors (torch.Tensor): Item latent factors matrix.
            interaction_matrix (torch.Tensor): Binary interaction matrix (1 if interaction, 0 otherwise).
            confidence_matrix (torch.Tensor): Confidence matrix (C_ui).

        Returns:
            torch.Tensor: The computed ALS loss.
        """

        # TODO: implement proper loss fonction for ALS

        # Predicted preference matrix
        prediction_matrix = torch.matmul(user_factors, item_factors.t())  # U * V^T

        # Compute the reconstruction error (squared error term)
        squared_error = confidence_matrix * (interaction_matrix - prediction_matrix) ** 2

        user_reg = user_factors.norm(p=2, dim=0)
        item_reg = item_factors.norm(p=2, dim=0)

        # Regularization term for user and item factors
        user_reg = torch.sum(user_factors ** 2)
        item_reg = torch.sum(item_factors ** 2)

        # ALS Loss: sum of squared error and regularization
        loss = torch.sum(squared_error) + self.reg_weight * (user_reg + item_reg)

        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
