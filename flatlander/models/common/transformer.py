import tensorflow as tf

from flatlander.models.common.attention import MultiHeadAttention


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dense_neurons, n_features,
                 n_actions, rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dense_neurons, n_features, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dense_neurons, n_features, rate)
        self.policy = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.softmax)
        self.value = tf.keras.layers.Dense(1)

    def call(self, input, target, train_mode, enc_padding_mask, look_ahead_mask, dec_padding_mask, positional_encoding):
        # + 1 since embeddings cannot have negative values
        enc_output = self.encoder(input+1, train_mode, enc_padding_mask, positional_encoding)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            target, enc_output, train_mode, look_ahead_mask, dec_padding_mask, positional_encoding)

        policy_out = self.policy(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        value_out = self.value(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return policy_out, value_out, attention_weights


class TransformerLayer(tf.keras.layers.Layer):

    @staticmethod
    def point_wise_feed_forward_network(d_model, dense_neurons):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dense_neurons, activation='relu'),  # (batch_size, seq_len, dense_neurons)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


class Encoder(TransformerLayer):
    def __init__(self, num_layers, d_model, num_heads, dense_neurons, n_features, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dense_neurons, rate)
                           for _ in range(num_layers)]

        self.embedding = tf.keras.layers.Embedding(n_features, d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, positional_encoding):

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += positional_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(TransformerLayer):
    def __init__(self, num_layers, d_model, num_heads, dense_neurons, n_features, rate=0.11):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(n_features, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dense_neurons, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask, positional_encoding):
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += positional_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class EncoderLayer(TransformerLayer):
    def __init__(self, d_model, num_heads, dense_neurons, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dense_neurons)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(TransformerLayer):
    def __init__(self, d_model, num_heads, dense_neurons, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = self.point_wise_feed_forward_network(d_model, dense_neurons)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
