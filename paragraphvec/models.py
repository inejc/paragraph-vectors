import torch
import torch.nn as nn


class DM(nn.Module):
    """Distributed Memory version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_docs: int
        Number of documents in a dataset.

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """
    def __init__(self, vec_dim, num_docs, num_words):
        super(DM, self).__init__()
        # paragraph matrix
        self._D = nn.Parameter(
            torch.randn(num_docs, vec_dim), requires_grad=True)
        # word matrix
        self._W = nn.Parameter(
            torch.randn(num_words, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

    def forward(self, context_ids, doc_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        context_ids: torch.Tensor of size (batch_size, num_context_words)
            Vocabulary indices of context words.

        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        # combine a paragraph vector with word vectors of
        # input (context) words
        x = torch.add(
            self._D[doc_ids, :], torch.sum(self._W[context_ids, :], dim=1))

        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        return torch.bmm(
            x.unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self, index):
        return self._D[index, :].data.tolist()


class DBOW(nn.Module):
    """Distributed Bag of Words version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_docs: int
        Number of documents in a dataset.

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """
    def __init__(self, vec_dim, num_docs, num_words):
        super(DBOW, self).__init__()
        # paragraph matrix
        self._D = nn.Parameter(
            torch.randn(num_docs, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

    def forward(self, doc_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        return torch.bmm(
            self._D[doc_ids, :].unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self, index):
        return self._D[index, :].data.tolist()
