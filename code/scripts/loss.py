import numpy as np

from mxnet.gluon.loss import Loss, SoftmaxCELoss

class SoftmaxCEMaskedLoss(SoftmaxCELoss):
    """Wrapper of the SoftmaxCELoss that supports valid_length as the input

    """
    def hybrid_forward(self, F, pred, label, valid_length): # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        F
        pred : Symbol or NDArray
            Shape (batch_size, length, V)
        label : Symbol or NDArray
            Shape (batch_size, length)
        valid_length : Symbol or NDArray
            Shape (batch_size, )
        Returns
        -------
        loss : Symbol or NDArray
            Shape (batch_size)
        """
        if self._sparse_label:
            sample_weight = F.cast(F.expand_dims(F.ones_like(label), axis=-1), dtype=np.float32)
        else:
            sample_weight = F.ones_like(label)

        sample_weight = F.SequenceMask(sample_weight,
                                       sequence_length=valid_length,
                                       use_sequence_length=True,
                                       axis=1)

        return super(SoftmaxCEMaskedLoss, self).hybrid_forward(F, pred, label, sample_weight)


class ICSLLoss(Loss):
    """Loss for IC/SL task.

    """

    def __init__(self, sparse_label=True, weight=None, batch_axis=0, **kwargs):  # pylint: disable=unused-argument
        super(ICSLLoss, self).__init__(
            weight=weight, batch_axis=batch_axis, **kwargs)
        self.ce_loss = SoftmaxCELoss()
        self.masked_ce_loss = SoftmaxCEMaskedLoss(sparse_label=sparse_label)

    def hybrid_forward(self, F, intent_pred, slot_pred, intent_label, slot_label, valid_length):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        intent_pred : intent prediction, shape (batch_size, num_intents)
        slot_pred : slot prediction, shape (batch_size, seq_length, num_slot_labels)
        intent_label : intent label, shape (batch_size)
        slot_label: slot label, shape (batch_size, seq_length)

        Returns
        -------
        outputs : NDArray
            Shape (batch_size)
        """
        intent_loss = self.ce_loss(intent_pred, intent_label)
        slot_loss = self.masked_ce_loss(slot_pred, slot_label, valid_length)
        return intent_loss + slot_loss
