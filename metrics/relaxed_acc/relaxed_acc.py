import re
import string
from typing import Any, Callable, Optional, Sequence
import datasets
import numpy as np

import evaluate

_DESCRIPTION = """
Returns the rate at which the input predicted strings exactly match their references, ignoring any strings input as part of the regexes_to_ignore list.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ExactMatch(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            reference_urls=[],
        )

    def _compute(
        self,
        predictions,
        references,
        max_relative_change: float = 0.05
    ):

        predictions = np.asarray(predictions)
        labels = np.asarray(references)
        
        def convert_to_float(x, num):
            if isinstance(x, str):
                if re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', x):
                    return float(x)
                else:
                    return num
            else:
                return float(x)
    
        def convert_to_word(x, word):
            if isinstance(x, str):
                if re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', x):
                    return word
                else:
                    return x
            else:
                return x  
        
        predictions_num = np.asarray([convert_to_float(p, 10.0) for p in predictions])
        labels_num = np.asarray([convert_to_float(l, 1.0) for l in labels])
        predictions_word = np.asarray([convert_to_word(p, 'a') for p in predictions])
        labels_word = np.asarray([convert_to_word(l, 'b') for l in labels])
        rel_num_change = np.abs((predictions_num - labels_num) / labels_num)
        score_num_list = rel_num_change  <= max_relative_change
        score_word_list = predictions_word == labels_word
        relaxed_acc = np.mean(score_num_list)+np.mean(score_word_list)


        return {"relaxed_acc": relaxed_acc}