"""Small regression checks for geometry calculations and rank/reveal inputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist

from rank import PROJECT, cluster_cloud, document_capped_rows, load_codes, load_groups, loop_scores, score_mode
from reveal import angular_summary, orient_circle


class GeometryTests(unittest.TestCase):
    def test_reconstruction_isometry(self):
        rng = np.random.default_rng(23)
        decoder = rng.normal(size=(6, 12))
        codes = np.maximum(rng.normal(size=(20, 6)), 0)
        features = [0, 2, 4]
        cloud, rows, _ = cluster_cloud(csc_matrix(codes), decoder, features, 20, rng, "uniform")
        expected = codes[rows][:, features] @ decoder[features]
        np.testing.assert_allclose(pdist(cloud), pdist(expected), atol=1e-10)

    def test_rank_input_does_not_require_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "00000.npz"
            second = Path(tmp) / "00001.npz"
            np.savez(first, values=[2.0, 1.0, 3.0], indices=[2, 0, 1], indptr=[0, 2, 3])
            np.savez(second, values=[4.0], indices=[1], indptr=[0, 0, 1])
            arr = load_codes([first, second], 3).toarray()
            np.testing.assert_array_equal(arr, [[1, 0, 2], [0, 3, 0], [0, 0, 0], [0, 4, 0]])

    def test_real_circle_and_empty_or_linear_cloud(self):
        theta = 2 * np.pi * np.arange(60) / 60
        ring = np.c_[np.cos(theta), np.sin(theta)]
        line = np.c_[np.linspace(-1, 1, 60), np.zeros(60)]
        self.assertGreater(loop_scores(ring)[0], 0.7)
        self.assertEqual(loop_scores(line)[0], 0.0)
        self.assertEqual(score_mode(np.empty((0, 3)), "shell"), (0.0, 0.0, 0.0))

    def test_document_cap_keeps_the_strongest_eligible_rows(self):
        codes = csc_matrix([[1.0, 0.0], [3.0, 0.0], [2.0, 1.0], [0.0, 0.0], [1.0, 2.0], [2.0, 2.0]])
        docs = np.array([3, 3, 3, 4, 7, 7], dtype=np.uint64)
        np.testing.assert_array_equal(document_capped_rows(codes, docs, 1), [1, 5])
        np.testing.assert_array_equal(document_capped_rows(codes, docs, 2), [1, 2, 4, 5])

    def test_rotation_reflection_and_translation_align(self):
        labels = np.arange(12)
        truth = np.exp(2j * np.pi * labels / 12)
        rotated = 3 + 2j + 2.5 * truth.conj() * np.exp(0.6j)
        origin, sign, phase = orient_circle(rotated, labels, 12)
        align = rotated - origin
        align = (align if sign == 1 else align.conj()) * phase
        metrics = angular_summary(align, labels, 12)
        self.assertEqual(metrics["token_accuracy"], 1.0)
        self.assertLess(metrics["centroid_mae_degrees"], 1e-8)

    def test_published_groups_partition_layer_7_sae(self):
        groups = load_groups(PROJECT / "published-groups.json")
        self.assertEqual(set(groups), set(range(1_000)))
        self.assertEqual(sorted(f for members in groups.values() for f in members), list(range(24_576)))
        self.assertEqual(groups[138], [2592, 4445, 4663, 4733, 6531, 8179, 9566, 20927, 24185])


if __name__ == "__main__":
    unittest.main()
