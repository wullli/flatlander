import os
import unittest

import numpy as np
import tensorflow as tf

from flatlander.models.tree_transformer import TreeTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TranformerTests(unittest.TestCase):

    def test_sample_strip(self):
        padded_obs_seq = np.full(shape=(21, 11), fill_value=-np.inf)
        padded_enc_seq = np.full(shape=(21, 15), fill_value=-np.inf)
        zeros_obs = np.zeros(shape=(5, 11))
        zeros_enc = np.zeros(shape=(5, 15))
        padded_obs_seq[:len(zeros_obs)] = zeros_obs
        padded_enc_seq[:len(zeros_enc)] = zeros_enc

        padded_enc_seq = tf.Variable(padded_enc_seq, dtype=float)
        padded_obs_seq = tf.Variable(padded_obs_seq, dtype=float)
        res_tensor = TreeTransformer.strip_sample((padded_obs_seq, padded_enc_seq))
        self.assertEqual(16, res_tensor[0].shape[0])
        self.assertEqual(16, res_tensor[1].shape[0])
