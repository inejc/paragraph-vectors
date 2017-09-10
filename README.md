## Paragraph Vectors
[![Build Status](https://travis-ci.org/inejc/paragraph-vectors.svg?branch=master)](https://travis-ci.org/inejc/paragraph-vectors)
[![codecov](https://codecov.io/gh/inejc/paragraph-vectors/branch/master/graph/badge.svg)](https://codecov.io/gh/inejc/paragraph-vectors)

A PyTorch implementation of Paragraph Vectors (doc2vec).

### Requirements
* [PyTorch](http://pytorch.org)
* [torchtext](https://github.com/pytorch/text) (available on PyPI)

### Usage
Put a csv file in the [data](data) directory (each row represents a single document and the first column should always contain the text).
```
data/example.csv
----------------
"In the week before their departure to Arrakis, when all the final scurrying about had reached a nearly unbearable frenzy, an old crone came to visit the mother of the boy, Paul.",...
"It was a warm night at Castle Caladan, and the ancient pile of stone that had served the Atreides family as home for twenty-six generations bore that cooled-sweat feeling it acquired before a change in the weather.",...
...
```

Todo.

### Resources
* [Distributed Representations of Words and Phrases and their Compositionality, T. Mikolov et al.](https://arxiv.org/abs/1310.4546)
* [Learning word embeddings efficiently with noise-contrastive estimation, A. Mnih et al.](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with)
* [Notes on Noise Contrastive Estimation and Negative Sampling, C. Dyer](https://arxiv.org/abs/1410.8251)
* [Approximating the Softmax (a blog post), S. Ruder](http://ruder.io/word-embeddings-softmax/index.html)
* [Distributed Representations of Sentences and Documents, Q. V. Le et al.](https://arxiv.org/abs/1405.4053)
* [Document Embedding with Paragraph Vectors, A. M. Dai et al.](https://arxiv.org/abs/1507.07998)
