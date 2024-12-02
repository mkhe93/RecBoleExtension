import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class ComputeSimilarity:
    def __init__(self, dataMatrix, topk=100, alpha=0.5, q=1.0):
        r"""Computes the asymmetric cosine similarity of dataMatrix with alpha and applies the weight adjustment using q.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink (int) : Hyper-parameter in cosine distance calculation.
            alpha (float):  Asymmetry control parameter in cosine similarity calculation.
            q (float): Weight adjustment exponent applied to the computed similarity.
        """

        super(ComputeSimilarity, self).__init__()

        self.alpha = alpha
        self.q = q

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topk, self.n_rows)  # Now working with users

        self.dataMatrix = dataMatrix.copy()

    def compute_similarity(self, method, block_size=100):
        r"""Compute the asymmetric cosine similarity for the given dataset, and adjust the weights using q.

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

        # Compute sum of interactions for normalization
        if method == "user":
            sumOfUsers = np.array(self.dataMatrix.sum(axis=1)).ravel()  # Sum of interactions for users
            end_local = self.n_rows
        else:
            raise NotImplementedError("Make sure 'method' is 'user'!")

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:
            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user
            data = self.dataMatrix[start_block:end_block, :]
            data = data.toarray()

            # Compute similarities between users
            this_block_weights = self.dataMatrix.dot(data.T)

            for index_in_block in range(this_block_size):
                this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0  # No self-similarity

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
                rows.extend(np.ones(numNotZeros) * Index)
                cols.extend(top_k_idx[notZerosMask])

            start_block += block_size

        W_sparse = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(self.n_rows, self.n_rows),
            dtype=np.float32,
        )
        return neigh, W_sparse.tocsc()


class AsymUserKNN(GeneralRecommender):
    r"""AsymUserKNN computes user similarity using asymmetric cosine similarity, applies weight adjustment with q, and uses the interaction matrix.
    Also applies the final score calculation with beta.
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(AsymUserKNN, self).__init__(config, dataset)

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
        ).compute_similarity("user")  # Compute user-to-user similarity
        self.pred_mat = self.w.dot(self.interaction_matrix).tolil()  # User-based prediction

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
        similar_users = np.argsort(self.w[user_id])[-self.k:]
        # Aggregate the similarity scores with normalization using beta
        score = 0
        norm_factor = len(similar_users) ** self.beta
        user_weights = np.zeros(len(similar_users))

        # Calculate weight norms for beta adjustment
        for i, user in enumerate(similar_users):
            if item_id in self.interaction_matrix[user].indices:
                weight = self.w[user_id, user]
                user_weights[i] = weight

        weight_norm = np.linalg.norm(user_weights) ** (2 * (1 - self.beta))

        if norm_factor > 0 and weight_norm > 0:
            score = np.sum(user_weights) / (norm_factor * weight_norm)

        return score
