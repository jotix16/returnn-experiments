

def segmental_blank_distribution(source, **kwargs):
  """ Caluclate distribution p(t_s | ..) [B,T'+1] over all time steps {t_{s-1}+1, ..., T}
      and does beam search.

  Args:
      source ([LayerBase]): label-emit and blank probabilities for each time step in {t_{s-1}+1, ..., T}
                            [emit_log_prob: [B,T',1], blank_log_prob: [B,T',1]] with T' = T - t_i+1
  Returns:
      [type]: p(t_s | ..) [B,T'+1]
  """
  import tensorflow as tf
  emit_log_prob = source(0, as_data=True, auto_convert=False)   # [B,T',1]
  emit_log_prob_t = tf.squeeze(emit_log_prob.get_placeholder_as_batch_major(), axis=-1)  # [B,T']
  blank_log_prob = source(1, as_data=True, auto_convert=False)  # [B,T',1]
  blank_log_prob_t = tf.squeeze(blank_log_prob.get_placeholder_as_batch_major(), axis=-1)  # [B,T']

  n_batch = emit_log_prob.get_batch_dim()
  max_t = emit_log_prob.time_dimension()  # T', which is T' = T - t_{s-1}+1

  # > Goal p(t_s | ..)
  # t_dist = [...] ; csum_blank + emit_prob

  # 1) Blanks during the segment
  # we compute for t= {t_{s-1}+1, ..., T} but since it is in log prod -> sum
  #       prod         (p_blank|a^{t_s-1}, h^T')
  # {t_{s-1}+1, ..., T}
  csum_blank = tf.concat([
    tf.tile([[0.]], [n_batch, 1]),  # for t_{s-1}+1 we have no contribution from the blank_probs (only emit) [B, 1]
    tf.cumsum(blank_log_prob_t, axis=1)  # inclusive [B, T]
  ], axis=1)  # [B,T'+1] where the first element is 0 for all

  # batchs: [B,T']                    times: [B,T']
  # [0,    0,  ..   0  ]              [0,  2,  .. T'-1]
  # [..,   .., ..   .. ]              [     ..        ]
  # [B-1,  B-1,  .. B-1]              [0,  2,  .. T'-1]
  batchs, times = tf.meshgrid(tf.range(n_batch), tf.range(max_t), indexing="ij")  # [B,T']
  batchs1, times1 = tf.meshgrid(tf.range(n_batch), tf.range(max_t + 1), indexing="ij")  # [B,T']

  # 2) Emit at the segment boundary
  # t_s = t_{i-1}+ 1..T' := p(emit)
  emit_prob_idxs = tf.stack([batchs, times], axis=-1)  # [B,T',2]
  emit_probs = tf.gather_nd(emit_log_prob_t, emit_prob_idxs)  # [B,T']

  emit_probs = tf.concat([emit_probs, tf.tile([[0.]], [n_batch, 1])], axis=1)  # [B,T'+1]
  mask = tf.greater_equal(times1, emit_log_prob.get_sequence_lengths()[:, None])
  emit_probs = tf.where(mask, tf.zeros_like(emit_probs), emit_probs)  # for the last item (t_i=T+1), we have no emission.

  t_dist = emit_probs + csum_blank  # [B,T'+1] distribution for t_s over all time steps {t_{s-1}+1, ..., T}
  from returnn.tf.util.basic import filter_ended_scores
  self = kwargs['self']
  search_choice = self._src_common_search_choices
  end_flags = self.network.get_rec_step_info().get_end_flag(search_choice)
  t_dist = filter_ended_scores(t_dist, end_flags, batch_dim=n_batch, dim=None)

  # print_op = tf.print("T'=", emit_log_prob.get_sequence_lengths()[0],
  #                     "blank_probs=", blank_log_prob_t[0], "\n\t",
  #                     "emit_log_prob_t=", emit_log_prob_t[0], "\n\t",
  #                     "mask=", mask[0], "\n\t",
  #                     "emit_probs=", emit_probs[0], "\n\t",
  #                     "csum_blank=", csum_blank[0], "\n\t",
  #                     "t_dist=", t_dist[0], "\n\t",
  #                     "sum_t_scores=", tf.exp(tf.reduce_logsumexp(t_dist, axis=1)),
  #                     summarize=-1)
  # with tf.control_dependencies([print_op]):
  #   t_dist = tf.identity(t_dist)
  return t_dist


# "alpha_t_s_label": {"class": "gather_nd", "from": "label_log_prob", "position": "rel_t_clip"},  # [B,K]
# "y_s_distribution": {"class": "eval", "from": ["alpha_t_s_label", "t", "base:encoder"], "eval": segmental_label_distribution,
def segmental_label_distribution(source, **kwargs):
  """ Caluclate p(y_s | ..) for the t_s-th segment

  Args:
      source ([LayerBase]): [alpha_t_s_label: [B,1030], t: [B], "encoder": [B, T, D]]
                            alpha_t_s_label: label distribution for segment ending at t_s
                            t: ts
                            encoder: only needed to get seq_len T TODO: not required

  Returns:
      [tf.Tensor]: scores is a [B,V=1030] matrix which holds the distribution over all labels
                   for each sequence in the batch for segment t_s. (V is the size of vocab)
  """
  from returnn.tf.compat import v1 as tf
  alpha_t_s_label = source(0, as_data=True, auto_convert=False)   # [B,V=1030]
  t = source(1, as_data=True, auto_convert=False)  # [B]
  t_s = t.get_placeholder_as_batch_major()  # [B]
  encoder = source(2, as_data=True, auto_convert=False)  # [B, T, D]
  alpha_t_s_label_t = alpha_t_s_label.get_placeholder_as_batch_major()  # (B, V=1030)

  n_batch = alpha_t_s_label.get_batch_dim()
  target_num_labels = alpha_t_s_label.dim

  zero_prob = float("-inf")
  # batchs: [B,V]                    vocab: [B,V=1030]
  # [0,    0,  .. 0   ]              [0,  1,  .. V-1]
  # [..,   .., .. ..  ]              [        ..    ]
  # [B-1,  B,  .. B-1 ]              [0,  1,  .. V-1]
  batchs, vocab = tf.meshgrid(tf.range(n_batch), tf.range(target_num_labels), indexing="ij")  # [B,K]

  # This assumes the following vocabulary:
  eos_symbol = 0  # TODO: not sure why 0 is the eos symbol
  # mask_eos_symbol: [B,V=1030]
  # [True,  False,  ..  False]
  # [True,  False,  ..  False]
  # [           ..           ]
  # [True,  False,  ..  False]
  mask_eos_symbol = tf.equal(vocab, eos_symbol)  # [B,V=1030] TODO: why is 0 eos
  mask_eos_time = tf.greater_equal(t_s, encoder.get_sequence_lengths())  # [B] t_s > T (eos)
  scores = tf.where(mask_eos_symbol,
                    tf.where(mask_eos_time,  # (B, V=1030) only for True value of mask_eos_symbol
                             tf.zeros_like(alpha_t_s_label_t),  # prob=1 for EOS, since t_s > T
                             tf.ones_like(alpha_t_s_label_t) * zero_prob  # prob=0 for t_s < T
                             ),
                    tf.where(mask_eos_time,   # (B, V=1030)
                             tf.ones_like(alpha_t_s_label_t) * zero_prob,  # 0 probability to emit if ts > T
                             alpha_t_s_label_t))  # scores for symbol != EOS, t_s < T

  # print_op = tf.print("t_s=", t_s, ", enc-T=", encoder.get_sequence_lengths(), "\n\t",
  #                     "scores=", scores[0, :10], "\n\t",
  #                     "scores_sum=", tf.reduce_logsumexp(scores[0, 1:]),
  #                     "scores_sum_exp=", tf.exp(tf.reduce_logsumexp(scores[0, 1:])),
  #                     "scores_sum_exp_all=", tf.exp(tf.reduce_logsumexp(scores[0])),
  #                     "scores_label_emit", tf.exp(tf.reduce_logsumexp(alpha_t_s_label_t)),
  #                     summarize=-1)
  # with tf.control_dependencies([print_op]):
  #   scores = tf.identity(scores)
  return scores
