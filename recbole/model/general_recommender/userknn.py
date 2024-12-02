import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class ComputeSimilarity:
    def __init__(self, dataMatrix, topk=100, normalize=True):
        r"""Computes the cosine similarity of dataMatrix.

        If it is computed on :math:`URM=|users| \times |items|`, pass the URM.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int): The k value in KNN.
            normalize (bool): If True, divide the dot product by the product of the norms.
        """

        super(ComputeSimilarity, self).__init__()
        self.normalize = normalize
        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topk, self.n_rows)  # We are computing user-to-user similarity

        self.dataMatrix = dataMatrix.copy()

    def compute_similarity(self, method, block_size=100):
        r"""Compute the similarity for the given dataset.

        Args:
            method (str): Must be 'user' for user similarity.
            block_size (int): Block size to divide matrix into smaller chunks for similarity computation.

        Returns:
            list: The similar nodes.
            scipy.sparse.csr_matrix: The similarity matrix.
        """

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        if method == "user":
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
            end_local = self.n_rows
        else:
            raise NotImplementedError("Only 'user' method is supported in UserKNN!")
        sumOfSquared = np.sqrt(sumOfSquared)

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:
            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user
            data = self.dataMatrix[start_block:end_block, :].toarray()

            # Compute user-to-user similarities
            this_block_weights = self.dataMatrix.dot(data.T)

            for index_in_block in range(this_block_size):
                this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0  # No self-similarity

                # Apply normalization, ensure denominator != 0
                if self.normalize:
                    denominator = sumOfSquared[Index] * sumOfSquared + 1e-6
                    this_line_weights = np.multiply(this_line_weights, 1 / denominator)

                # Sort indices and select TopK
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_partition_sorting = np.argsort(-this_line_weights[relevant_partition])
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_line_weights[top_k_idx][notZerosMask])
                rows.extend(np.ones(numNotZeros) * Index)
                cols.extend(top_k_idx[notZerosMask])

            start_block += block_size

        # Build sparse matrix
        W_sparse = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(self.n_rows, self.n_rows),
            dtype=np.float32,
        )
        return neigh, W_sparse.tocsc()


class UserKNN(GeneralRecommender):
    r"""UserKNN is a basic model that computes user similarity with the interaction matrix."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(UserKNN, self).__init__(config, dataset)

        # Load parameters
        self.k = config["k"]

        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]

        # Compute user-user similarity
        _, self.w = ComputeSimilarity(
            self.interaction_matrix, topk=self.k
        ).compute_similarity("user")

        # User-based predictions
        self.pred_mat = self.w.dot(self.interaction_matrix).tolil()

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
        """Calculate the final score for a user-item pair using the neighbors' weighted ratings."""
        similar_users = np.argsort(self.w[user_id])[-self.k:]
        numerator = 0
        denominator = 0

        # Calculate weighted sum of neighbors' ratings
        for v in similar_users:
            weight = self.w[user_id, v]
            rating = self.interaction_matrix[v, item_id]

            if rating != 0:  # Only consider users who have rated the item
                numerator += weight * rating
                denominator += weight

        return numerator / denominator if denominator > 0 else 0
