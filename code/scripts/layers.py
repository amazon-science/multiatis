import mxnet as mx

from gluonnlp.model.attention_cell import AttentionCell, _masked_softmax
from gluonnlp.model.block import L2Normalization
from gluonnlp.model.transformer import PositionwiseFFN
from mxnet.gluon import nn, HybridBlock

class ScaledDotProductAttentionCell(AttentionCell):
    """Dot product attention between the query and the key.

    Depending on parameters, defined as::

        units is None:
            score = <h_q, h_k>
        units is not None and luong_style is False:
            score = <W_q h_q, W_k h_k>
        units is not None and luong_style is True:
            score = <W h_q, h_k>

    Parameters
    ----------
    units: int or None, default None
        Project the query and key to vectors with `units` dimension
        before applying the attention. If set to None,
        the query vector and the key vector are directly used to compute the attention and
        should have the same dimension::

            If the units is None,
                score = <h_q, h_k>
            Else if the units is not None and luong_style is False:
                score = <W_q h_q, W_k, h_k>
            Else if the units is not None and luong_style is True:
                score = <W h_q, h_k>

    luong_style: bool, default False
        If turned on, the score will be::

            score = <W h_q, h_k>

        `units` must be the same as the dimension of the key vector
    scaled: bool, default True
        Whether to divide the attention weights by the sqrt of the query dimension.
        This is first proposed in "[NIPS2017] Attention is all you need."::

            score = <h_q, h_k> / sqrt(dim_q)

    normalized: bool, default False
        If turned on, the cosine distance is used, i.e::

            score = <h_q / ||h_q||, h_k / ||h_k||>

    use_bias : bool, default True
        Whether to use bias in the projection layers.
    dropout : float, default 0.0
        Attention dropout
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias
    prefix : str or None, default None
        See document of `Block`.
    params : str or None, default None
        See document of `Block`.
    """

    def __init__(self, units=None, luong_style=False, scaled=True, normalized=False, use_bias=True,
                 dropout=0.0, temperature=1.0, weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(ScaledDotProductAttentionCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._scaled = scaled
        self._normalized = normalized
        self._use_bias = use_bias
        self._luong_style = luong_style
        self._dropout = dropout
        self._temperature = temperature
        if self._luong_style:
            assert units is not None, 'Luong style attention is not available without explicitly ' \
                                      'setting the units'
        with self.name_scope():
            self._dropout_layer = nn.Dropout(dropout)
        if units is not None:
            with self.name_scope():
                self._proj_query = nn.Dense(units=self._units, use_bias=self._use_bias,
                                            flatten=False, weight_initializer=weight_initializer,
                                            bias_initializer=bias_initializer, prefix='query_')
                if not self._luong_style:
                    self._proj_key = nn.Dense(units=self._units, use_bias=self._use_bias,
                                              flatten=False, weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer, prefix='key_')
        if self._normalized:
            with self.name_scope():
                self._l2_norm = L2Normalization(axis=-1)

    def _compute_weight(self, F, query, key, mask=None):
        if self._units is not None:
            query = self._proj_query(query)
            if not self._luong_style:
                key = self._proj_key(key)
            elif F == mx.nd:
                assert query.shape[-1] == key.shape[-1], 'Luong style attention requires key to ' \
                                                         'have the same dim as the projected ' \
                                                         'query. Received key {}, query {}.'.format(
                    key.shape, query.shape)
        if self._normalized:
            query = self._l2_norm(query)
            key = self._l2_norm(key)
        if self._scaled:
            query = F.contrib.div_sqrt_dim(query)

        att_score = F.batch_dot(query, key, transpose_b=True) / self._temperature

        att_weights = self._dropout_layer(_masked_softmax(F, att_score, mask, self._dtype))
        return att_weights


class AttentionMapCell(HybridBlock):
    """Structure of the Transformer Decoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
        Number of units for the output
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, units=128, hidden_size=512, dropout=0.0, use_residual=True,
                 attn_temperature=1.0, weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(AttentionMapCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._dropout = dropout
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.attention_cell = ScaledDotProductAttentionCell(temperature=attn_temperature,
                                                                scaled=True,
                                                                normalized=False)
            self.proj_layer = nn.Dense(units=units, flatten=False,
                                       use_bias=False,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       prefix='proj_inter_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size,
                                       units=units,
                                       use_residual=use_residual,
                                       dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)

            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, mem_value, mem_mask=None):  #pylint: disable=unused-argument
        #  pylint: disable=arguments-differ
        """Transformer Decoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mem_value : Symbol or NDArrays
            Memory value, i.e. output of the encoder. Shape (batch_size, mem_length, C_in)
        mem_mask : Symbol or NDArray or None
            Mask for mem_value. Shape (batch_size, length, mem_length)

        Returns
        -------
        decoder_cell_outputs: list
            Outputs of the decoder cell. Contains:

            - outputs of the transformer decoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer decoder cell
        """
        attention_outputs, attention_weights = \
            self.attention_cell(inputs, mem_value, mem_value, mem_mask)
        outputs = self.proj_layer(attention_outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        return outputs, attention_outputs
