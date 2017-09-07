import torch.nn as nn


class DistributedMemory(nn.Module):
    """Distributed Memory version of Paragraph Vectors."""

    def __init__(self):
        super(DistributedMemory, self).__init__()

    def forward(self):
        raise NotImplementedError()


class DistributedBagOfWords(nn.Module):
    """Distributed Bag of Words version of Paragraph Vectors."""

    def __init__(self):
        super(DistributedBagOfWords, self).__init__()

    def forward(self):
        raise NotImplementedError()
