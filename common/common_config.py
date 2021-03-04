
"""
Provides some common settings for RETURNN configs (training + search).

Use `from ... import *` directly in your config for this module.
"""

from .multi_gpu_horovod import *  # noqa
from returnn.config import get_global_config

config = get_global_config()

# task
use_tensorflow = True
task = config.value("task", "train")
device = "gpu"

debug_mode = False
if int(os.environ.get("RETURNN_DEBUG", "0")):
  import sys
  print("** DEBUG MODE", file=sys.stderr)
  debug_mode = True

if config.has("beam_size"):
  beam_size = config.int("beam_size", 0)
  import sys
  print("** beam_size %i" % beam_size, file=sys.stderr)
else:
  beam_size = 12

search_output_layer = "decision"
debug_print_layer_output_template = True  # doesn't cost anything. always recommended

log_batch_size = True  # only relevant with verbosity 5
tf_log_memory_usage = True

# Verbosity 5 will print some stats (loss, batch size, maybe other things) per single minibatch.
# All lower verbosity will not. (If you don't want 5, then recommended would be 4 or 3.)
# (You will get an interactive bar for verbosity<5 for the epoch progress if you use an interactive shell.)
log_verbosity = 5