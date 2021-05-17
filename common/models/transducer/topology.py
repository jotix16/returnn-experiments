from .loss import (rnnt_loss, rnnt_loss_out_type,
                   rnnt_tf_loss, rna_tf_loss, rna_loss_out_type)
from .alignment import (rnnt_alignment, rnnt_alignment_out_type,
                        rna_alignment, rna_alignment_out_type)
from typing import Callable, Dict, Any


class Topology:
  """
  Hold informations about different label topologies such as loss-, alignment-funcion and their out_types.

  The loss, alignment_out_type and alignment function all are to be used in EvalLayers
  which take as inputs the following layers:
        output_log_prob: [B, T, U+1, V] log-probabilities
        target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  where
        B: batch, T: time, U:target/labels, V: vocabulary, U': seq_len of alignment
  EvalLayer offers a source() callback, which has to be used to get the mentioned data.
  """
  def __init__(self,
               name: str,
               loss,
               loss_out_type,
               alignment,
               alignment_out_type):
    """ Label Topology such as rnnt, rna, ctc.
    Args:
        loss: function (source: (i: int, as_data: bool = False, ...) -> tf.Tensor|Data, ...) -> tf.Tensor[B]
        loss_out_type: function (...) -> Data[B]
        alignment: function (source: (i: int, as_data: bool = False, ...) -> tf.Tensor|Data, ...) -> tf.Tensor[B,U']
        alignment_out_type: function (sources: list[LayerBase], ...) -> Data[B,U']
    """
    self.name = name
    self.loss = loss
    self.loss_out_type = loss_out_type
    self.alignment = alignment
    self.alignment_out_type = alignment_out_type

  def make(self, output_emit: str, encoder: str) -> Dict[str, Any]:
    """
      Args:
          output_emit (str): layer providing True if a non blank label is emited
      Returns:
          Dict[str, Any]: dict that provides the position in the input and target seq "t" and "u"
                          together with "end" layer if topology is not time-synchronous.
      """
    top_dict = {
      "const0": {"class": "constant", "value": 0, "collocate_with": ["du", "dt", "t", "u"], "dtype": "int32"},
      "const1": {"class": "constant", "value": 1, "collocate_with": ["du", "dt", "t", "u"], "dtype": "int32"},
      # pos in target, [B]
      "u": {"class": "combine", "from": ["prev:u", "du"], "kind": "add", "initial_output": 0},
      # pos in input, [B]
      "t": {"class": "combine", "from": ["dt", "prev:t"], "kind": "add", "initial_output": 0},
    }
    top_dict.update(self._make(output_emit=output_emit, encoder=encoder))
    return top_dict

  def _make(self, output_emit: str, encoder: str) -> Dict[str, Any]:
    raise NotImplementedError


class RnaTopology(Topology):
  def _make(self, output_emit: str, encoder: str) -> Dict[str, Any]:
    return {
      "du": {"class": "switch", "condition": output_emit, "true_from": "const1", "false_from": "const0"},
      "dt": {"class": "switch", "condition": output_emit, "true_from": "const1", "false_from": "const0"}
      # no "end" required as it is time synchronous
    }


class RnntTopology(Topology):
  def _make(self, output_emit: str, encoder: str) -> Dict[str, Any]:
    return {
      "du": {"class": "switch", "condition": output_emit, "true_from": "const1", "false_from": "const0"},
      "dt": {"class": "switch", "condition": output_emit, "true_from": "const0", "false_from": "const1"},
      # stop at U+T
      # in recog: stop when all input has been consumed
      # in train: defined by target.
      "enc_seq_len": {"class": "length", "from": encoder, "sparse": False},
      "end": {"class": "compare", "from": ["t", "enc_seq_len"], "kind": "equal"}
    }


rna_topology = RnaTopology(
  name="rna",
  loss=rna_tf_loss,
  loss_out_type=rna_loss_out_type,
  alignment=rna_alignment,
  alignment_out_type=rna_alignment_out_type)

rnnt_topology_tf = RnntTopology(
  name="rnnt_tf",
  loss=rnnt_tf_loss,
  loss_out_type=rnnt_loss_out_type,
  alignment=rnnt_alignment,
  alignment_out_type=rnnt_alignment_out_type)

rnnt_topology = RnntTopology(
  name="rnnt",
  loss=rnnt_loss,
  loss_out_type=rnnt_loss_out_type,
  alignment=rnnt_alignment,
  alignment_out_type=rnnt_alignment_out_type)
