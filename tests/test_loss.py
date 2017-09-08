from unittest import TestCase

from paragraphvec.loss import NegativeSampling


class NegativeSamplingTest(TestCase):

    def setUp(self):
        self.loss = NegativeSampling()

    def test_forward(self):
        with self.assertRaises(NotImplementedError):
            self.loss.forward()
