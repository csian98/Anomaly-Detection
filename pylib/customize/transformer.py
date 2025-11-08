from framework.base_sequential import BaseSequential

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

import tensorflow as tf
import keras.layers as layers
from keras.layers import Dense, Layer, MultiHeadAttention, Dropout, LayerNormalization, Conv1D

# Transformer Models

# Decoder Block
# ref: https://github.com/liamdm/FlowTransformer/blob/master/implementations/transformers/basic/decoder_block.py
class TransformerDecoderBlock(Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=input_dimension)
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(inner_dimension, activation='relu'),
            Dense(input_dimension)
        ])
        self.dropout2 = Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    # noinspection PyMethodOverriding
    # SIAN
    def call(self, inputs, training=True, mask=None):
        # inputs = (target_seq, enc_output)
        target_seq = inputs
        enc_output = inputs

        # self attention of target_seq
        attn_output = self.mha(target_seq, target_seq)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = target_seq + attn_output
        out1 = self.layernorm1(out1)

        # multi-head attention with encoder output as the key and value, and target_seq as the query
        attn_output = self.mha(out1, enc_output)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = out1 + attn_output
        out2 = self.layernorm2(out2)

        # feed forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out3 = out2 + ffn_output
        out3 = self.layernorm2(out3)

        return out3

# Encoder Block
# ref: https://github.com/liamdm/FlowTransformer/blob/master/implementations/transformers/basic/encoder_block.py
class GPT3Attention(layers.Layer):
    def __init__(self, n_heads, d_model, dropout_rate=0.1):
        super(GPT3Attention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        output = self.dropout(output)

        return output

class MultiHeadAttentionImplementation:
    Keras = 0,
    GPT3 = 1

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1, use_conv:bool=False, prefix:str=None, attn_implementation:MultiHeadAttentionImplementation = MultiHeadAttentionImplementation.Keras):

        if prefix is None:
            prefix = ""

        super().__init__(name=f"{prefix}transformer_encoder")

        if inner_dimension < input_dimension:
            warnings.warn(f"Typically inner_dimension should be greater than or equal to the input_dimension!")

        self.attn_implementation = attn_implementation

        self.dropout_rate = dropout_rate
        self.attention = \
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension, name=f"{prefix}multi_head_attn") \
                if attn_implementation == MultiHeadAttentionImplementation.Keras else\
                GPT3Attention(num_heads, inner_dimension, dropout_rate=0.0)

        layer_norm = 1e-6

        self.attention_dropout = layers.Dropout(dropout_rate, name=f"{prefix}attention_dropout")
        self.attention_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}attention_layer_norm")

        self.feed_forward_0 = Conv1D(filters=inner_dimension, kernel_size=1, activation="relu", name=f"{prefix}feed_forward_0") \
            if use_conv else Dense(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0")
        self.feed_forward_1 = Conv1D(filters=input_dimension, kernel_size=1, activation="relu", name=f"{prefix}feed_forward_1") \
            if use_conv else Dense(input_dimension, activation="relu", name=f"{prefix}feed_forward_1")

        self.feed_forward_dropout = layers.Dropout(dropout_rate, name=f"{prefix}feed_forward_dropout")
        self.feed_forward_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}feed_forward_layer_norm")

    # noinspection PyMethodOverriding
    # SIAN
    def call(self, inputs, training=True, mask=None):
        x = inputs
        x = self.attention(x, x) if self.attn_implementation == MultiHeadAttentionImplementation.Keras else self.attention(x, x, x, mask)

        attention_output = self.attention_dropout(x, training=training) if self.dropout_rate > 0 else x

        x = inputs + attention_output
        x = self.attention_layer_norm(x)
        x = self.feed_forward_0(x)
        x = self.feed_forward_1(x)
        x = self.feed_forward_dropout(x, training=training) if self.dropout_rate > 0 else x
        feed_forward_output = x

        return self.feed_forward_layer_norm(attention_output + feed_forward_output)

# Basic Transformers
# ref: https://github.com/liamdm/FlowTransformer/blob/master/implementations/transformers/basic_transformers.py
class BasicTransformer(BaseSequential):

    @property
    def name(self) -> str:
        if self.use_conv:
            return f"Basic Conv Transformer" + (" Decoder" if self.is_decoder else "")
        else:
            return f"Basic Dense Transformer" + (" Decoder" if self.is_decoder else "")

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "use_conv": self.use_conv,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.internal_size
        }

    def __init__(self, n_layers:int, internal_size:int, n_heads:int, use_conv:bool=False, dropout_rate:float=0.1, is_decoder=False):
        super().__init__()
        self.n_layers = n_layers
        self.internal_size = internal_size
        self.use_conv = use_conv
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.is_decoder = is_decoder

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            if self.is_decoder:
                if self.use_conv:
                    raise NotImplementedError()
                m_x = TransformerDecoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate)(m_x)
            else:
                m_x = TransformerEncoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate, use_conv=self.use_conv, prefix=f"{prefix}block_{layer_i}_")(m_x)

        return m_x

# Named Transformers
# ref: https://github.com/liamdm/FlowTransformer/blob/master/implementations/transformers/named_transformers.py
class GPTSmallTransformer(BaseSequential):

    @property
    def name(self) -> str:
        return "GPT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 12
        self.internal_size = 768
        self.n_heads = 12
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = True

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            m_x = TransformerDecoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate)(m_x)

        return m_x


class BERTSmallTransformer(BaseSequential):

    @property
    def name(self) -> str:
        return "BERT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 12
        self.internal_size = 768
        self.n_heads = 12
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = False

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            m_x = TransformerEncoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate, prefix=f"block_{layer_i}_")(m_x)

        return m_x
