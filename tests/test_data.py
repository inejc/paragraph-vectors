from unittest import TestCase

from paragraphvec.data import load_dataset, NCEData


class NCEDataTest(TestCase):

    def setUp(self):
        self.dataset = load_dataset('example.csv')

    def test_num_examples_for_different_batch_sizes(self):
        len_1 = self._num_examples_with_batch_size(1)

        for batch_size in range(2, 100):
            len_x = self._num_examples_with_batch_size(batch_size)
            self.assertEqual(len_x, len_1)

    def _num_examples_with_batch_size(self, batch_size):
        nce_data = NCEData(
            self.dataset,
            batch_size=batch_size,
            context_size=2,
            num_noise_words=3,
            max_size=1,
            num_workers=1)
        num_batches = len(nce_data)
        nce_data.start()
        nce_generator = nce_data.get_generator()

        total = 0
        for _ in range(num_batches):
            batch = next(nce_generator)
            total += len(batch)
        nce_data.stop()
        return total

    def test_multiple_iterations(self):
        nce_data = NCEData(
            self.dataset,
            batch_size=16,
            context_size=3,
            num_noise_words=3,
            max_size=1,
            num_workers=1)
        num_batches = len(nce_data)
        nce_data.start()
        nce_generator = nce_data.get_generator()

        iter0_targets = []
        for _ in range(num_batches):
            batch = next(nce_generator)
            iter0_targets.append([x[0] for x in batch.target_noise_ids])

        iter1_targets = []
        for _ in range(num_batches):
            batch = next(nce_generator)
            iter1_targets.append([x[0] for x in batch.target_noise_ids])

        for ts0, ts1 in zip(iter0_targets, iter1_targets):
            for t0, t1 in zip(ts0, ts0):
                self.assertEqual(t0, t1)
        nce_data.stop()

    def test_different_batch_sizes(self):
        nce_data = NCEData(
            self.dataset,
            batch_size=16,
            context_size=1,
            num_noise_words=3,
            max_size=1,
            num_workers=1)
        num_batches = len(nce_data)
        nce_data.start()
        nce_generator = nce_data.get_generator()

        targets0 = []
        for _ in range(num_batches):
            batch = next(nce_generator)
            for ts in batch.target_noise_ids:
                targets0.append(ts[0])
        nce_data.stop()

        nce_data = NCEData(
            self.dataset,
            batch_size=19,
            context_size=1,
            num_noise_words=3,
            max_size=1,
            num_workers=1)
        num_batches = len(nce_data)
        nce_data.start()
        nce_generator = nce_data.get_generator()

        targets1 = []
        for _ in range(num_batches):
            batch = next(nce_generator)
            for ts in batch.target_noise_ids:
                targets1.append(ts[0])
        nce_data.stop()

        for t0, t1 in zip(targets0, targets1):
            self.assertEqual(t0, t1)

    def test_tensor_sizes(self):
        nce_data = NCEData(
            self.dataset,
            batch_size=32,
            context_size=5,
            num_noise_words=3,
            max_size=1,
            num_workers=1)
        nce_data.start()
        nce_generator = nce_data.get_generator()
        batch = next(nce_generator)
        nce_data.stop()

        self.assertEqual(batch.context_ids.size()[0], 32)
        self.assertEqual(batch.context_ids.size()[1], 10)
        self.assertEqual(batch.doc_ids.size()[0], 32)
        self.assertEqual(batch.target_noise_ids.size()[0], 32)
        self.assertEqual(batch.target_noise_ids.size()[1], 4)

    def test_parallel(self):
        nce_data = NCEData(
            self.dataset,
            batch_size=32,
            context_size=5,
            num_noise_words=1,
            max_size=1,
            num_workers=2)
        nce_data.start()
        nce_generator = nce_data.get_generator()
        batch00 = next(nce_generator)
        batch01 = next(nce_generator)
        batch02 = next(nce_generator)
        nce_data.stop()

        nce_data = NCEData(
            self.dataset,
            batch_size=32,
            context_size=5,
            num_noise_words=1,
            max_size=1,
            num_workers=2)
        nce_data.start()
        nce_generator = nce_data.get_generator()
        batch10 = next(nce_generator)
        batch11 = next(nce_generator)
        batch12 = next(nce_generator)
        nce_data.stop()

        batch0_same = any([
            self.batches_same(batch00, batch10),
            self.batches_same(batch00, batch11),
            self.batches_same(batch00, batch12)
        ])
        batch1_same = any([
            self.batches_same(batch01, batch10),
            self.batches_same(batch01, batch11),
            self.batches_same(batch01, batch12)
        ])
        batch2_same = any([
            self.batches_same(batch02, batch10),
            self.batches_same(batch02, batch11),
            self.batches_same(batch02, batch12)
        ])
        self.assertTrue(batch0_same and batch1_same and batch2_same)

    @staticmethod
    def batches_same(batch0, batch1):
        for t0, t1 in zip(batch0.target_noise_ids, batch1.target_noise_ids):
            if t0[0] != t1[0]:
                return False
        return True


class DataUtilsTest(TestCase):

    def setUp(self):
        self.dataset = load_dataset('example.csv')

    def test_load_dataset(self):
        self.assertEqual(len(self.dataset), 4)

    def test_vocab(self):
        self.assertTrue(self.dataset.fields['text'].use_vocab)
        self.assertTrue(len(self.dataset.fields['text'].vocab) > 0)
