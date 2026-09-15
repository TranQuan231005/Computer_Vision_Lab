"""Đối chứng độc lập cho phần đánh giá và băm — Thanh Nguyên."""
import unittest
import numpy as np
from PIL import Image
from vision_wavelet import (BaselineHasher, classification_metrics, evaluate_distances,
                            split_source_pairs, benchmark_configurations)


class EvaluationTests(unittest.TestCase):
    def test_confusion_counts(self):
        m = classification_metrics([1, 1, 0, 0], [0, 2, 1, 3], 1)
        self.assertEqual([m[k] for k in ('tp', 'tn', 'fp', 'fn')], [1, 1, 1, 1])
        for key in ('accuracy', 'recall', 'specificity', 'precision', 'f1'):
            self.assertEqual(m[key], .5)
        self.assertEqual(classification_metrics([0], [1], 0)['precision'], 0)

    def test_auc_ties_and_endpoints(self):
        for distances, auc in (([0, 0, 4, 4], 1), ([4, 4, 0, 0], 0), ([2, 2, 2, 2], .5)):
            rows = [dict(label=y, distance=d) for y, d in zip([1, 1, 0, 0], distances)]
            result = evaluate_distances(rows, 4)
            self.assertEqual(result['auc'], auc)
            self.assertEqual(result['roc']['fpr'][0], 0)
            self.assertEqual(result['roc']['tpr'][-1], 1)
        self.assertEqual(evaluate_distances([dict(label=1, distance=0), dict(label=0, distance=4)], 4)['optimal']['threshold'], 0)
        for rows in ([dict(label=1, distance=0)], [dict(label=1, distance=-1), dict(label=0, distance=2)]):
            with self.assertRaises(ValueError):
                evaluate_distances(rows, 4)

    def test_source_split(self):
        rows = [dict(source_id1=str(i), source_id2=str(j)) for i in range(9) for j in range(i, 9)]
        a, b, split = split_source_pairs(rows)
        self.assertFalse(set(split['calibration_sources']) & set(split['test_sources']))
        for pairs, key in ((a, 'calibration_sources'), (b, 'test_sources')):
            for pair in pairs:
                self.assertIn(pair['source_id1'], split[key])
                self.assertIn(pair['source_id2'], split[key])
        self.assertEqual(len(a) + len(b) + split['excluded_cross_pairs'], len(rows))
        self.assertEqual(split_source_pairs(rows), (a, b, split))

    def test_baselines(self):
        gradient = np.tile(np.arange(9, dtype=np.uint8) * 25, (8, 1))
        self.assertTrue(BaselineHasher('dhash').hash(gradient).all())
        self.assertFalse(BaselineHasher('dhash').hash(gradient[:, ::-1]).any())
        self.assertTrue(BaselineHasher('ahash').hash(Image.new('L', (8, 8), 128)).all())
        h = BaselineHasher('phash')
        np.testing.assert_allclose(h._dct @ h._dct.T, np.eye(32), atol=1e-14)
        for hasher in benchmark_configurations().values():
            bits = hasher.hash(gradient)
            self.assertEqual(bits.size, hasher.n_bits)
            self.assertEqual(bits.dtype, np.dtype(bool))


if __name__ == '__main__':
    unittest.main(verbosity=2)
