"""Kiểm thử tính đúng của lõi wHash và hợp đồng dữ liệu — Trần Ngọc Nhân."""
import json
import unittest

import numpy as np
from PIL import Image
import pywt

from vision_wavelet import (ROOT, WaveletHasher, augment_image, hash_to_hex,
                            hamming_distance, preprocess_image, similarity_percentage)


class WaveletTests(unittest.TestCase):
    """Dùng kết quả giải tích và phép nghịch đảo làm đối chứng độc lập."""

    def test_known_haar(self):
        image = np.repeat(np.repeat(np.array([[0, 64], [128, 255]], dtype=np.uint8), 2, 0), 2, 1)
        result = WaveletHasher(2, 4, level=1).analyze(image)
        np.testing.assert_allclose(result['LL'], np.array([[0, 128], [256, 510]]) / 255)
        for key in ('LH', 'HL', 'HH'):
            np.testing.assert_allclose(result[key], 0, atol=1e-14)
        np.testing.assert_array_equal(result['bits'], [0, 0, 1, 1])
        self.assertEqual(result['hex'], '3')

    def test_reconstruction_and_sizes(self):
        image = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        for wave in ('haar', 'db2', 'db4', 'sym4', 'bior2.2', 'coif2'):
            for size in (8, 16):
                for threshold in ('median', 'mean'):
                    hasher = WaveletHasher(size, 64, wave, 2, threshold)
                    result = hasher.analyze(image)
                    restored = pywt.waverec2(result['coeffs'], wave, mode='periodization')
                    np.testing.assert_allclose(restored, result['gray'], atol=1e-10)
                    self.assertEqual(result['bits'].size, size * size)
                    self.assertEqual(hamming_distance(result['bits'], hasher.hash(image)), 0)

    def test_hamming(self):
        self.assertEqual(hamming_distance('0101', '0011'), 2)
        self.assertEqual(similarity_percentage('0101', '0011'), 50)
        self.assertEqual(hash_to_hex([0] * 64), '0000000000000000')
        self.assertEqual(hamming_distance([0] * 256, [1] * 256), 256)
        for a, b in [('', ''), ([0], [1, 0]), ([0, 2], [0, 1]), ('ab', '01')]:
            with self.assertRaises(ValueError):
                hamming_distance(a, b)

    def test_invalid_configuration(self):
        for args in [dict(hash_size=7), dict(level=0), dict(wavelet='coif2', level=6),
                     dict(hash_size=16), dict(threshold='other')]:
            with self.assertRaises(ValueError):
                WaveletHasher(**args)

    def test_transparency_and_uniform_image(self):
        transparent = Image.new('RGBA', (15, 10), (0, 0, 0, 0))
        np.testing.assert_allclose(preprocess_image(transparent), 1)
        for value in (0, 255):
            bits = WaveletHasher().hash(Image.new('L', (32, 32), value))
            self.assertTrue(bits.all())

    def test_reproducible_augmentation(self):
        image = np.full((40, 50, 3), 128, dtype=np.uint8)
        first, second = list(augment_image(image, 9)), list(augment_image(image, 9))
        self.assertEqual(len(first), 15)
        for (name, a, params), (other, b, other_params) in zip(first, second):
            self.assertEqual((name, params), (other, other_params))
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(np.asarray(first[8][1]), np.asarray(list(augment_image(image, 10))[8][1])))

    def test_dataset_contract(self):
        manifest = json.loads((ROOT / 'data/dataset_pairs.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['originals']), 15)
        pairs = manifest['pairs']
        self.assertEqual(sum(p['label'] for p in pairs), 225)
        self.assertEqual(len(pairs), 330)
        seen = set()
        for pair in pairs:
            key = tuple(sorted((pair['image1'], pair['image2'])))
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(pair['source_id1'] == pair['source_id2'], bool(pair['label']))
            for path in key:
                absolute = (ROOT / 'data' / path).resolve()
                self.assertTrue(absolute.is_relative_to((ROOT / 'data').resolve()))
                with Image.open(absolute) as image:
                    image.verify()


if __name__ == '__main__':
    unittest.main(verbosity=2)
