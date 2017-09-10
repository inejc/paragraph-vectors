from unittest import TestCase

import torch

from paragraphvec.loss import NegativeSampling


class NegativeSamplingTest(TestCase):

    def setUp(self):
        self.loss_f = NegativeSampling()

    def test_forward(self):
        # todo: test actual value
        scores = torch.FloatTensor([[12.1, 1.3, 6.5], [18.9, 2.1, 9.4]])
        loss = self.loss_f.forward(scores)
        self.assertTrue(loss.data[0] >= 0)
