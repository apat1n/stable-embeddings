import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LightGCNDataset(Dataset):
    def __init__(self, user_items: csr_matrix, n_negatives: int = 10, verify_negative_samples: bool = True):
        self.n_negatives = n_negatives
        self.verify_negative_samples = verify_negative_samples

        self.n_users = user_items.shape[0]
        self.m_items = user_items.shape[1]
        n_nodes = self.n_users + self.m_items

        user_items_coo = user_items.tocoo()
        self.unique_users = user_items_coo.row

        tmp_adj = sp.csr_matrix((user_items_coo.data, (user_items_coo.row, user_items_coo.col + self.n_users)),
                                shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # normalize matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        # normalize by user counts
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        # normalize by item counts
        normalized_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # convert to torch sparse matrix
        adj_mat_coo = normalized_adj_matrix.tocoo()

        values = adj_mat_coo.data
        indices = np.vstack((adj_mat_coo.row, adj_mat_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj_mat_coo.shape

        self.adj_matrix = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    def get_user_positives(self, user):
        return self.adj_matrix[user].coalesce().indices()[0] - self.n_users

    def get_user_negatives(self, user, k=10):
        neg = []
        positives = set(self.get_user_positives(user)) if self.verify_negative_samples else []
        while len(neg) < k:
            candidate = np.random.randint(1, self.m_items)
            if not self.verify_negative_samples or \
                    self.verify_negative_samples and candidate not in positives:
                neg.append(candidate)
        return neg

    def get_sparse_graph(self):
        """
        Returns a graph in torch.sparse_coo_tensor.
        A = |0,   R|
            |R^T, 0|
        """
        return self.adj_matrix

    def __len__(self):
        return len(self.unique_users)

    def __getitem__(self, idx):
        """
        returns user, pos_items, neg_items

        :param idx: index of user from unique_users
        :return:
        """
        user = self.unique_users[idx]
        pos = np.random.choice(self.get_user_positives(user), self.n_negatives)
        neg = self.get_user_negatives(user, self.n_negatives)
        return user, pos, neg


def collate_function(batch):
    users = []
    pos_items = []
    neg_items = []
    for user, pos, neg in batch:
        users.extend([user for _ in pos])
        pos_items.extend(pos)
        neg_items.extend(neg)
    return list(map(torch.tensor, [users, pos_items, neg_items]))


class LightGCN(nn.Module):
    def __init__(
            self,
            regularization: float = 0.01,
            batch_size: int = 128,
            factors: int = 100,
            n_negatives: int = 10,
            iterations: int = 100,
            n_layers: int = 2,
            verify_negative_samples: bool = True
    ):
        """
        :param regularization: float, optional
            The regularization factor to use
        :param batch_size: int, optional
            Size of the batch used in training
        :param factors: int, optional
            The number of latent factors to compute
        :param n_negatives: int, optional
            The number of negative candidates in sampling
        :param iterations: int, optional
            The number of training epochs to use when fitting the data
        verify_negative_samples: bool, optional
            When sampling negative items, check if the randomly picked negative item has actually
            been liked by the user. This check increases the time needed to train but usually leads
            to better predictions.
        """
        super(LightGCN, self).__init__()
        self.regularization = regularization
        self.batch_size = batch_size
        self.factors = factors
        self.n_negatives = n_negatives
        self.iterations = iterations
        self.n_layers = n_layers
        self.verify_negative_samples = verify_negative_samples

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def __init_weight(self, dataset: LightGCNDataset):
        """
        Initialize embeddings with normal distribution
        :return:
        """
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=dataset.n_users, embedding_dim=self.factors).to(self.device)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=dataset.m_items, embedding_dim=self.factors).to(self.device)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.Graph = dataset.get_sparse_graph().to(self.device)

    def computer(self) -> tuple:
        """
        Propagate high-hop embeddings for lightGCN
        :return: user embeddings, item embeddings
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        layer_embeddings = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            layer_embeddings.append(all_emb)
        layer_embeddings = torch.stack(layer_embeddings, dim=1)

        final_embeddings = layer_embeddings.mean(dim=1)  # output is mean of all layers
        users, items = torch.split(final_embeddings, [self.num_users, self.num_items])
        return users, items

    def get_embedding(self, users: torch.tensor, pos_items: torch.tensor,
                      neg_items: torch.tensor) -> tuple:
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users: torch.tensor, pos: torch.tensor, neg: torch.tensor) -> tuple:
        """
        Calculate BPR loss as - sum ln(sigma(pos_scores - neg_scores)) + L2 norm
        :param users: users for which calculate loss
        :param pos: positive items
        :param neg: negative items
        :return: loss, reg_loss
        """
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        roc_auc = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())

        loss = - (pos_scores - neg_scores).sigmoid().log().mean()
        return loss, reg_loss, roc_auc

    def forward(self, users: torch.tensor, items: torch.tensor):
        # compute embedding
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_prod = torch.mul(users_emb, items_emb)
        return torch.sum(inner_prod, dim=1).sigmoid()

    def fit(self, user_items: csr_matrix, callback_fn=None):
        """
        Fitting model with BPR loss.
        :param user_items: dataset for training
        :param callback_fn: callback function
        :return:
        """
        pbar = tqdm(range(self.iterations))

        dataset = LightGCNDataset(user_items, self.n_negatives, self.verify_negative_samples)
        self.__init_weight(dataset)

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_function)

        optimizer = torch.optim.Adam(self.parameters())
        for _ in pbar:
            for users, pos, neg in dataloader:
                optimizer.zero_grad()
                users, pos, neg = users.to(self.device), pos.to(self.device), neg.to(self.device)
                loss, reg_loss, roc_auc = self.bpr_loss(users, pos, neg)
                total_loss = loss + self.regularization * reg_loss

                total_loss.backward()
                optimizer.step()

                pbar.set_postfix({'bpr_loss': total_loss.item(), 'train_auc': roc_auc})
