## Paragraph Vectors
[![Build Status](https://travis-ci.org/inejc/paragraph-vectors.svg?branch=master)](https://travis-ci.org/inejc/paragraph-vectors)
[![codecov](https://codecov.io/gh/inejc/paragraph-vectors/branch/master/graph/badge.svg)](https://codecov.io/gh/inejc/paragraph-vectors)
[![codebeat badge](https://codebeat.co/badges/e5008ad0-240c-48e9-a158-2547989b798e)](https://codebeat.co/projects/github-com-inejc-paragraph-vectors-master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c865067aa4194184ae0c649b865b1fd2)](https://www.codacy.com/app/inejc/paragraph-vectors?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=inejc/paragraph-vectors&amp;utm_campaign=Badge_Grade)

A PyTorch implementation of Paragraph Vectors (doc2vec).
<p align="center">
    <img src="/.github/dmdbow.png?raw=true"/>
</p>

All models minimize the Negative Sampling objective as proposed by T. Mikolov et al. [1]. This provides scope for sparse updates (i.e. only vectors of sampled noise words are used in forward and backward passes). In addition to that, batches of training data (with noise sampling) are generated in parallel on a CPU while the model is trained on a GPU.

**Caveat emptor!** Be warned that **`paragraph-vectors`** is in an early-stage development phase. Feedback, comments, suggestions, contributions, etc. are more than welcome.

### Installation
1. Install [PyTorch](http://pytorch.org) (follow the link for instructions).
2. Install the **`paragraph-vectors`** library.
```
git clone https://github.com/inejc/paragraph-vectors.git
cd paragraph-vectors
pip install -e .
```
Note that installation in a virtual environment is the recommended way.

### Usage
1. Put a csv file in the [data](data) directory. Each row represents a single document and the first column should always contain the text. Note that a header line is mandatory.
```text
data/example.csv
----------------
text,...
"In the week before their departure to Arrakis, when all the final scurrying about had reached a nearly unbearable frenzy, an old crone came to visit the mother of the boy, Paul.",...
"It was a warm night at Castle Caladan, and the ancient pile of stone that had served the Atreides family as home for twenty-six generations bore that cooled-sweat feeling it acquired before a change in the weather.",...
...
```
2. Run [train.py](paragraphvec/train.py) with selected parameters (models are saved in the [models](models) directory).
```bash
python train.py start --data_file_name 'example.csv' --num_epochs 100 --batch_size 32 --num_noise_words 2 --vec_dim 100 --lr 1e-3
```

#### Parameters
* **`data_file_name`**: str\
Name of a file in the *data* directory.
* **`model_ver`**: str, one of ('dm', 'dbow'), default='dbow'\
Version of the model as proposed by Q. V. Le et al. [5], Distributed Representations of Sentences and Documents. 'dbow' stands for Distributed Bag Of Words, 'dm' stands for Distributed Memory.
* **`vec_combine_method`**: str, one of ('sum', 'concat'), default='sum'\
Method for combining paragraph and word vectors when model_ver='dm'. Currently only the 'sum' operation is implemented.
* **`context_size`**: int, default=0\
Half the size of a neighbourhood of target words when model_ver='dm' (i.e. how many words left and right are regarded as context). When model_ver='dm' context_size has to greater than 0, when model_ver='dbow' context_size has to be 0.
* **`num_noise_words`**: int\
Number of noise words to sample from the noise distribution.
* **`vec_dim`**: int\
Dimensionality of vectors to be learned (for paragraphs and words).
* **`num_epochs`**: int\
Number of iterations to train the model (i.e. number of times every example is seen during training).
* **`batch_size`**: int\
Number of examples per single gradient update.
* **`lr`**: float\
Learning rate of the Adam optimizer.
* **`save_all`**: bool, default=False\
Indicates whether a checkpoint is saved after each epoch. If false, only the best performing model is saved.
* **`generate_plot`**: bool, default=True\
Indicates whether a diagnostic plot displaying loss value over epochs is generated after each epoch.
* **`max_generated_batches`**: int, default=5\
Maximum number of pre-generated batches.
* **`num_workers`**: int, default=1\
Number of batch generator jobs to run in parallel. If value is set to -1, total number of machine CPUs is used. Note that order of batches is not guaranteed when **`num_workers`** > 1.

3. Export trained paragraph vectors to a csv file (vectors are saved in the [data](data) directory).
```bash
python export_vectors.py start --data_file_name 'example.csv' --model_file_name 'example_model.dbow_numnoisewords.2_vecdim.100_batchsize.32_lr.0.001000_epoch.25_loss.0.981524.pth.tar'
```

#### Parameters
* **`data_file_name`**: str\
Name of a file in the *data* directory that was used during training.
* **`model_file_name`**: str\
Name of a file in the *models* directory (a model trained on the **`data_file_name`** dataset).

### Example of trained vectors
First two principal components (1% cumulative variance explained) of 300-dimensional document vectors trained on arXiv abstracts. Shown are two subcategories from Computer Science. Dataset was comprised of 74219 documents and 91417 unique words.
<p align="center">
    <img src="/.github/learned_vectors_pca.png?raw=true"/>
</p>

### Resources
* [1] [Distributed Representations of Words and Phrases and their Compositionality, T. Mikolov et al.](https://arxiv.org/abs/1310.4546)
* [2] [Learning word embeddings efficiently with noise-contrastive estimation, A. Mnih et al.](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with)
* [3] [Notes on Noise Contrastive Estimation and Negative Sampling, C. Dyer](https://arxiv.org/abs/1410.8251)
* [4] [Approximating the Softmax (a blog post), S. Ruder](http://ruder.io/word-embeddings-softmax/index.html)
* [5] [Distributed Representations of Sentences and Documents, Q. V. Le et al.](https://arxiv.org/abs/1405.4053)
* [6] [Document Embedding with Paragraph Vectors, A. M. Dai et al.](https://arxiv.org/abs/1507.07998)
