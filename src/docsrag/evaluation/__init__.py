"""Evaluation harness.

Split into three concerns:

* ``dataset``  -- the eval set format and IO.
* ``retrieval`` -- sweep retrieval configurations and score them with IR metrics.
* ``faithfulness`` -- LLM-as-judge grading of whether answers are supported by
  their retrieved context.

Retrieval evaluation needs no API key at all, which means you can iterate on
chunking and fusion for free and fast. Only faithfulness grading costs tokens.
"""

from .dataset import EvalExample, load_eval_set, save_eval_set
from .retrieval import RetrievalReport, evaluate_retrieval, sweep

__all__ = (
    "EvalExample",
    "load_eval_set",
    "save_eval_set",
    "RetrievalReport",
    "evaluate_retrieval",
    "sweep",
)
