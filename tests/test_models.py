from unittest import TestCase

from paragraphvec.models import DistributedMemory, DistributedBagOfWords


class DistributedMemoryTest(TestCase):

    def setUp(self):
        self.model = DistributedMemory()

    def test_forward(self):
        with self.assertRaises(NotImplementedError):
            self.model.forward()


class DistributedBagOfWordsTest(TestCase):

    def setUp(self):
        self.model = DistributedBagOfWords()

    def test_forward(self):
        with self.assertRaises(NotImplementedError):
            self.model.forward()
