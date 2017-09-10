from unittest import TestCase

from paragraphvec.data import load_dataset, NCEGenerator


class NCEIteratorTest(TestCase):

    def setUp(self):
        dataset = load_dataset('example.csv')
        self.iter = NCEGenerator(
            dataset, batch_size=4, context_size=2, num_noise_words=3)

    def test_iteration(self):
        # todo
        # next(iter(self.iter))
        pass


class DataUtilsTest(TestCase):

    def setUp(self):
        self.dataset = load_dataset('example.csv')

    def test_load_dataset(self):
        self.assertEqual(len(self.dataset), 4)

    def test_vocab(self):
        self.assertTrue(self.dataset.fields['text'].use_vocab)
        self.assertTrue(len(self.dataset.fields['text'].vocab) > 0)
