import torch
import torch.nn as nn


class DistributedMemory(nn.Module):
    """Distributed Memory version of Paragraph Vectors."""

    def __init__(self, vec_dim, num_docs, num_words):
        super(DistributedMemory, self).__init__()
        # paragraph matrix
        self._D = nn.Parameter(
            torch.rand(num_docs, vec_dim), requires_grad=True)
        # word matrix
        self._W = nn.Parameter(
            torch.rand(num_words, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.rand(vec_dim, num_words), requires_grad=True)

    def forward(self, context_ids, doc_ids, target_noise_ids):
        """Todo."""
        # combine a paragraph vector with word vectors of
        # input (context) words
        x = torch.add(
            self._D[doc_ids, :], torch.sum(self._W[context_ids, :], dim=1))

        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        return torch.bmm(
            x.unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze()


class DistributedBagOfWords(nn.Module):
    """Distributed Bag of Words version of Paragraph Vectors."""

    def __init__(self):
        super(DistributedBagOfWords, self).__init__()

    def forward(self):
        raise NotImplementedError()
