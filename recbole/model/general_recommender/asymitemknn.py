import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class ComputeSimilarity:
    def __init__(self, dataMatrix, topk=100, alpha=0.5, q=1.0):
        r"""Computes the asymmetric cosine similarity of dataMatrix with alpha parameter.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            alpha (float):  Asymmetry control parameter in cosine similarity calculation.
        """

        super(ComputeSimilarity, self).__init__()

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topk, self.n_columns)
        self.alpha = alpha
        self.q = q

        self.dataMatrix = dataMatrix.copy()

    def compute_similarity(self, method, block_size=100):
        r"""Compute the asymmetric cosine similarity for the given dataset.

        Args:
            method (str) : Calculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
            block_size (int): Divide matrix into blocks for efficient calculation.

        Returns:
            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num],
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        if method == "user":
            sumOfUsers = np.array(self.dataMatrix.sum(axis=1)).ravel()
            end_local = self.n_rows
        elif method == "item":
            sumOfUsers = np.array(self.dataMatrix.sum(axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError("Make sure 'method' is in ['user', 'item']!")

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:
            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user or item
            if method == "user":
                data = self.dataMatrix[start_block:end_block, :]
            else:
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray()

            # Compute similarities
            if method == "user":
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)

            for index_in_block in range(this_block_size):
                this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0

                # Apply asymmetric cosine normalization and shrinkage
                denominator = (
                        (sumOfUsers[Index] ** self.alpha) *
                        (sumOfUsers ** (1 - self.alpha)) + 1e-6
                )
                this_line_weights = np.divide(this_line_weights, denominator)

                # Apply weight adjustment via q: f(w) = w^q
                this_line_weights = np.power(this_line_weights, self.q)

                # Sort indices and select TopK
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_partition_sorting = np.argsort(-this_line_weights[relevant_partition])
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_line_weights[top_k_idx][notZerosMask])
                if method == "user":
                    rows.extend(np.ones(numNotZeros) * Index)
                    cols.extend(top_k_idx[notZerosMask])
                else:
                    rows.extend(top_k_idx[notZerosMask])
                    cols.extend(np.ones(numNotZeros) * Index)

            start_block += block_size

        # End while
        if method == "user":
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)),
                shape=(self.n_rows, self.n_rows),
                dtype=np.float32,
            )
        else:
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)),
                shape=(self.n_columns, self.n_columns),
                dtype=np.float32,
            )
        return neigh, W_sparse.tocsc()


class AsymItemKNN(GeneralRecommender):
    r"""AsymItemKNN computes item similarity using asymmetric cosine similarity and interaction matrix."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(AsymItemKNN, self).__init__(config, dataset)

        # load parameters info
        self.k = config["k"]
        self.alpha = config['alpha'] if 'alpha' in config else 0.5  # Asymmetric cosine parameter
        self.q = config['q'] if 'q' in config else 1.0  # Weight adjustment exponent
        self.beta = config['beta'] if 'beta' in config else 0.5  # Beta for final score normalization

        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        _, self.w = ComputeSimilarity(
            self.interaction_matrix, topk=self.k, alpha=self.alpha, q=self.q
        ).compute_similarity("item")
        self.pred_mat = self.interaction_matrix.dot(self.w).tolil()

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["w", "pred_mat"]

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)
        result = []

        for index in range(len(user)):
            uid = user[index]
            iid = item[index]
            score = self.calculate_final_score(uid, iid)
            result.append(score)
        result = torch.from_numpy(np.array(result)).to(self.device)
        return result

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()

        score = self.pred_mat[user, :].toarray().flatten()
        result = torch.from_numpy(score).to(self.device)

        return result

    def calculate_final_score(self, user_id, item_id):
        """Calculate the final score using the normalized form with beta."""
        similar_items = np.argsort(self.w[item_id])[-self.k:]
        # Aggregate the similarity scores with normalization using beta
        score = 0
        norm_factor = len(similar_items) ** self.beta
        item_weights = np.zeros(len(similar_items))

        # Calculate weight norms for beta adjustment
        for i, item in enumerate(similar_items):
            if item in self.interaction_matrix[user_id].indices:
                weight = self.w[item_id, item]
                item_weights[i] = weight

        weight_norm = np.linalg.norm(item_weights) ** (2 * (1 - self.beta))

        if norm_factor > 0 and weight_norm > 0:
            score = np.sum(item_weights) / (norm_factor * weight_norm)

        return score