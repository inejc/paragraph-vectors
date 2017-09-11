from unittest import TestCase

from paragraphvec.data import load_dataset, NCEGenerator


class NCEIteratorTest(TestCase):

    def setUp(self):
        self.dataset = load_dataset('example.csv')

    def test_num_examples_for_different_batch_sizes(self):
        len_1 = self._num_examples_with_batch_size(1)
        for batch_size in range(2, 33):
            len_x = self._num_examples_with_batch_size(batch_size)
            self.assertEqual(len_x, len_1)

    def _num_examples_with_batch_size(self, batch_size):
        nce_generator = NCEGenerator(
            self.dataset,
            batch_size=batch_size,
            context_size=2,
            num_noise_words=3)

        total = 0
        for _ in range(len(nce_generator)):
            batch = nce_generator.next()
            total += len(batch)
        return total

    def test_multiple_iterations(self):
        nce_generator = NCEGenerator(
            self.dataset,
            batch_size=16,
            context_size=3,
            num_noise_words=3)

        iter0_targets = []
        for _ in range(len(nce_generator)):
            batch = nce_generator.next()
            iter0_targets.append([x[0] for x in batch.target_noise_ids])

        iter1_targets = []
        for _ in range(len(nce_generator)):
            batch = nce_generator.next()
            iter1_targets.append([x[0] for x in batch.target_noise_ids])

        for ts0, ts1 in zip(iter0_targets, iter1_targets):
            for t0, t1 in zip(ts0, ts0):
                self.assertEqual(t0, t1)

    def test_different_batch_sizes(self):
        nce_generator = NCEGenerator(
            self.dataset,
            batch_size=16,
            context_size=1,
            num_noise_words=3)

        targets0 = []
        for _ in range(len(nce_generator)):
            batch = nce_generator.next()
            for ts in batch.target_noise_ids:
                targets0.append(ts[0])

        nce_generator = NCEGenerator(
            self.dataset,
            batch_size=19,
            context_size=1,
            num_noise_words=3)

        targets1 = []
        for _ in range(len(nce_generator)):
            batch = nce_generator.next()
            for ts in batch.target_noise_ids:
                targets1.append(ts[0])

        for t0, t1 in zip(targets0, targets1):
            self.assertEqual(t0, t1)

    def test_tensor_sizes(self):
        nce_generator = NCEGenerator(
            self.dataset,
            batch_size=32,
            context_size=5,
            num_noise_words=3)
        batch = nce_generator.next()

        self.assertEqual(batch.context_ids.size()[0], 32)
        self.assertEqual(batch.context_ids.size()[1], 10)
        self.assertEqual(batch.doc_ids.size()[0], 32)
        self.assertEqual(batch.target_noise_ids.size()[0], 32)
        self.assertEqual(batch.target_noise_ids.size()[1], 4)


class DataUtilsTest(TestCase):

    def setUp(self):
        self.dataset = load_dataset('example.csv')

    def test_load_dataset(self):
        self.assertEqual(len(self.dataset), 4)

    def test_vocab(self):
        self.assertTrue(self.dataset.fields['text'].use_vocab)
        self.assertTrue(len(self.dataset.fields['text'].vocab) > 0)
