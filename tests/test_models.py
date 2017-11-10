from unittest import TestCase

import torch

from paragraphvec.loss import NegativeSampling
from paragraphvec.models import DM, DBOW


class DMTest(TestCase):

    def setUp(self):
        self.batch_size = 2
        self.num_noise_words = 2
        self.num_docs = 3
        self.num_words = 15
        self.vec_dim = 10

        self.context_ids = torch.LongTensor([[0, 2, 5, 6], [3, 4, 1, 6]])
        self.doc_ids = torch.LongTensor([1, 2])
        self.target_noise_ids = torch.LongTensor([[1, 3, 4], [2, 4, 7]])
        self.model = DM(
            self.vec_dim, self.num_docs, self.num_words)

    def test_num_parameters(self):
        self.assertEqual(
            sum([x.size()[0] * x.size()[1] for x in self.model.parameters()]),
            self.num_docs * self.vec_dim + 2 * self.num_words * self.vec_dim)

    def test_forward(self):
        x = self.model.forward(
            self.context_ids, self.doc_ids, self.target_noise_ids)

        self.assertEqual(x.size()[0], self.batch_size)
        self.assertEqual(x.size()[1], self.num_noise_words + 1)

    def test_backward(self):
        cost_func = NegativeSampling()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        for _ in range(2):
            x = self.model.forward(
                self.context_ids, self.doc_ids, self.target_noise_ids)
            x = cost_func.forward(x)
            self.model.zero_grad()
            x.backward()
            optimizer.step()

        self.assertEqual(torch.sum(self.model._D.grad[0, :].data), 0)
        self.assertNotEqual(torch.sum(self.model._D.grad[1, :].data), 0)
        self.assertNotEqual(torch.sum(self.model._D.grad[2, :].data), 0)

        context_ids = self.context_ids.numpy().flatten()
        target_noise_ids = self.target_noise_ids.numpy().flatten()

        for word_id in range(15):
            if word_id in context_ids:
                self.assertNotEqual(
                    torch.sum(self.model._W.grad[word_id, :].data), 0)
            else:
                self.assertEqual(
                    torch.sum(self.model._W.grad[word_id, :].data), 0)

            if word_id in target_noise_ids:
                self.assertNotEqual(
                    torch.sum(self.model._O.grad[:, word_id].data), 0)
            else:
                self.assertEqual(
                    torch.sum(self.model._O.grad[:, word_id].data), 0)


class DBOWTest(TestCase):

    def setUp(self):
        self.batch_size = 2
        self.num_noise_words = 2
        self.num_docs = 3
        self.num_words = 15
        self.vec_dim = 10

        self.doc_ids = torch.LongTensor([1, 2])
        self.target_noise_ids = torch.LongTensor([[1, 3, 4], [2, 4, 7]])
        self.model = DBOW(
            self.vec_dim, self.num_docs, self.num_words)

    def test_num_parameters(self):
        self.assertEqual(
            sum([x.size()[0] * x.size()[1] for x in self.model.parameters()]),
            self.num_docs * self.vec_dim + self.num_words * self.vec_dim)

    def test_forward(self):
        x = self.model.forward(self.doc_ids, self.target_noise_ids)

        self.assertEqual(x.size()[0], self.batch_size)
        self.assertEqual(x.size()[1], self.num_noise_words + 1)

    def test_backward(self):
        cost_func = NegativeSampling()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        for _ in range(2):
            x = self.model.forward(self.doc_ids, self.target_noise_ids)
            x = cost_func.forward(x)
            self.model.zero_grad()
            x.backward()
            optimizer.step()

        self.assertEqual(torch.sum(self.model._D.grad[0, :].data), 0)
        self.assertNotEqual(torch.sum(self.model._D.grad[1, :].data), 0)
        self.assertNotEqual(torch.sum(self.model._D.grad[2, :].data), 0)

        target_noise_ids = self.target_noise_ids.numpy().flatten()

        for word_id in range(15):
            if word_id in target_noise_ids:
                self.assertNotEqual(
                    torch.sum(self.model._O.grad[:, word_id].data), 0)
            else:
                self.assertEqual(
                    torch.sum(self.model._O.grad[:, word_id].data), 0)
