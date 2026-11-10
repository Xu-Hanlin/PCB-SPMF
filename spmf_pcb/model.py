"""Paper name for the exact, checkpoint-compatible SPMF implementation.

The third-party class identifier is retained solely for existing PyTorch files.
No layer, tensor name, parameter, or mathematical operation is changed by naming.
"""

from ultralytics.nn.modules.DEA import DEA as SPMF

__all__ = ["SPMF"]
