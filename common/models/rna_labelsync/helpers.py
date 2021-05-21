

# "prev_output": {"class": "eval", "from": ["prev:output", "am_window"], "eval": prev_output},
def prev_output(source, **kwargs):
  """ For each seq in batch it returns
      [prev:output 1030 1030 1030 ...] (length T')
  Args:
      source ([LayerBase]): prev_output [B], am_window [B, T', D] atention window over encoder

  Returns:
      [type]:      t_{s-1}     ...       T          of size     [B, T']
               1  [a1_{s-1},   <b>,  .. <b>]
               .. [  ...       ...   ..    ]
               B  [aB_{s-1},   <b>,  .. <b>]
  """

  from returnn.tf.compat import v1 as tf
  prev_output = source(0, as_data=True, auto_convert=False)   # [B] -> [1030]
  am_window = source(1, as_data=True, auto_convert=False)  # [B, T', D]
  t_prime = am_window.time_dimension()
  n_batch = am_window.get_batch_dim()

  # [B, 1] ; [B, T'-1]
  prev_output_t = tf.expand_dims(prev_output.get_placeholder_as_batch_major(), axis=1)
  blank_ones = tf.ones((n_batch, t_prime - 1), dtype=tf.int32) * targetb_blank_idx  # TODO
  out = tf.concat([prev_output_t, blank_ones], axis=1)  # [B,T']
  return out


def search_checks(layer):
    """
    :param ChoiceLayer layer:
    :rtype: list[tf.Operation]
    """
    # if task != "train":
    #   return []
    if not layer.search_choices:  # training without search
        return []
    from returnn.tf.compat import v1 as tf
    from TFUtil import get_shape
    from TFNetworkLayer import SelectSearchSourcesLayer
    checks = []
    # without any recombination, we expect all beam entries to be non-inf
    beam_scores = layer.search_choices.beam_scores  # (batch,beam)
    i = layer.network.get_rec_step_index()
    n_batch, n_beam = get_shape(beam_scores)
    label = tf.reshape(layer.output.placeholder, (n_batch, n_beam))  # (batch,beam)
    in_scores = tf.reshape(SelectSearchSourcesLayer.select_if_needed(
      layer.sources[0], search_choices=layer.search_choices).output.placeholder,
      (n_batch, n_beam, -1))  # (batch,beam,dim)
    if layer.name == "rel_t":
        t_layer = layer
    else:
        t_layer = SelectSearchSourcesLayer.select_if_needed(
          layer=layer.network.get_layer("rel_t"),
          search_choices=layer.search_choices)
    t = tf.reshape(t_layer.output.placeholder, (n_batch, n_beam))  # (batch,beam)
    prev_t_layer = SelectSearchSourcesLayer.select_if_needed(
      layer=layer.network.get_layer("prev:t"),
      search_choices=layer.search_choices)
    prev_t = tf.reshape(prev_t_layer.output.placeholder, (n_batch, n_beam))  # (batch,beam)
    # t_base_seq = layer.network.get_layer("base:data:t_base").output.placeholder  # (batch,dec-T)
    # t_base = layer.network.get_layer("data:t_base").output.placeholder  # (batch,)
    enc_seq_lens = layer.network.get_layer("base:encoder").output.get_sequence_lengths()  # (batch,)
    dec_seq_lens = layer.network.get_layer("base:data:%s" % target).output.get_sequence_lengths()  # (batch,)
    end_flags = tf.reshape(layer.network.get_rec_step_info().get_end_flag(
      target_search_choices=layer.search_choices), (n_batch, n_beam))  # (batch,beam)
    beam_scores_finite = tf.is_finite(beam_scores)
    beam_scores_any_finite = tf.reduce_all(tf.logical_or(beam_scores_finite, end_flags), axis=1)  # (batch,)
    # beam_scores_cheat_finite = beam_scores_finite[:,-1]  # (batch,). the last beam is the cheating beam
    t_check = tf.logical_or(end_flags, tf.greater(t, prev_t))  # (batch,beam)
    t_check = tf.logical_or(t_check, tf.equal(t, enc_seq_lens[:,None] - 1))  # allow to loop in last frame
    t_check = tf.logical_and(tf.reduce_any(t_check, axis=1), t_check[:,-1])  # (batch,)
    bad_batch_idx = tf.argmin(tf.cast(
      tf.logical_and(beam_scores_any_finite, t_check), tf.int32))
    checks.append(tf.print({"beam_scores_finite": tf.logical_or(beam_scores_finite, end_flags)}, summarize=-1))
    checks.append(
      tf.Assert(
        # tf.logical_and(
        #   tf.logical_and(tf.reduce_all(beam_scores_any_finite), tf.reduce_all(beam_scores_cheat_finite)),
        #   tf.reduce_all(beam_scores_any_finite),
        #   tf.reduce_all(t_check)),
        ["layer", layer.name, "i", i,
          "any_finite", beam_scores_any_finite,# "cheat_finite", beam_scores_cheat_finite,
          "bad_batch_idx", bad_batch_idx, "n_batch", n_batch, "n_beam", n_beam,
          "beam scores", beam_scores[bad_batch_idx],
          "label", label[bad_batch_idx],
          "prev:t", prev_t[bad_batch_idx],
          "enc_seq_len", enc_seq_lens[bad_batch_idx],
          "dec_seq_len", dec_seq_lens[bad_batch_idx],
          "end_flag", end_flags[bad_batch_idx],
          "scores_in_", in_scores[bad_batch_idx],  # in_scores[bad_batch_idx,-1],
          # "scores_in", layer.search_scores_in[bad_batch_idx, -1],
          "scores_base", layer.search_scores_base[bad_batch_idx],
          "scores_comb", layer.search_scores_combined[bad_batch_idx, -1]],
        summarize=100,
        name="beam_scores_finite_check"))
    return checks
