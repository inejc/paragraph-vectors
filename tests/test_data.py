from unittest import TestCase

from paragraphvec.data import load_dataset, NCEIterator


class NCEIteratorTest(TestCase):

    def setUp(self):
        dataset = load_dataset('example.csv')
        self.iter = NCEIterator(
            dataset, batch_size=4, context_size=1, num_noise_words=3)

    def test_return_value(self):
        pass


class DataUtilsTest(TestCase):

    def test_load_dataset(self):
        dataset = load_dataset('example.csv')
        self.assertEqual(len(dataset), 4)
