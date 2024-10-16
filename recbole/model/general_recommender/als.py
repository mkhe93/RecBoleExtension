# -*- coding: utf-8 -*-
# @Time   : 2024/10/08
# @Author : Your Name
# @Email  : your_email@example.com

r"""
ALS
################################################
Reference:
    Hu, Y., Koren, Y., & Volinsky, C. (2008). "Collaborative Filtering for Implicit Feedback Datasets." In ICDM 2008.
"""

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType, ModelType
from recbole.model.loss import ALSLoss


class ALS(GeneralRecommender):
    r"""ALS is a matrix factorization model that optimizes the embeddings using Alternating Least Squares."""

    input_type = InputType.POINTWISE
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(ALS, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.LABEL = config['LABEL_FIELD']


        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = nn.MSELoss()
        self.loss = ALSLoss()
        self.sigmoid = nn.Sigmoid()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        # loss = self.loss(output, label)
        loss = self.loss(self.get_user_embedding(user), self.get_item_embedding(item), label, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
