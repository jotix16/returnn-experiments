
import tensorflow as tf
import sys


def tf_debug_print_start_end():  # -------------------------------
  sess = tf.compat.v1.Session()
  with sess.as_default():
    print_op_start_end = tf.print("""
                                  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                  """, "",
                                  output_stream=sys.stdout)
  sess.run(print_op_start_end)


def tf_debug_print(*inputs, **kargs):  # -------------------------------
  sess = tf.compat.v1.Session()
  with sess.as_default():
    print_op = tf.print(*inputs, **kargs)
    sess.run(print_op)
