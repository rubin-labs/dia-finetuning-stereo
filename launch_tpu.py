
import os
import sys
import torch_xla.distributed.xla_multiprocessing as xmp
from dia.train_acc_tpu import main

def _mp_fn(index, args):
    sys.argv = [sys.argv[0]] + args
    main()

if __name__ == '__main__':
    # Pass all arguments after the script name to the training function
    args = sys.argv[1:]
    xmp.spawn(_mp_fn, args=(args,), start_method='fork')

