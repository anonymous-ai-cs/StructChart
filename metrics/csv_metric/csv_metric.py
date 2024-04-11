import re
import string
from typing import Any, Callable, Optional, Sequence
import datasets
import numpy as np
import Levenshtein

import evaluate

_DESCRIPTION = """
Returns the rate at which the input predicted strings exactly match their references, ignoring any strings input as part of the regexes_to_ignore list.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CSVMetric(evaluate.Metric):
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
    ):

        predictions = np.asarray(predictions)
        labels = np.asarray(references)
        def is_int(val):
            try:
                int(val)
                return True
            except ValueError:
                return False

        def is_float(val):
            try:
                float(val)
                return True
            except ValueError:
                return False
        
        # def csv2triples(csv, separator='\\t', delimiter='\\n'):
        #     lines = csv.strip().split(delimiter)
        #     header = lines[0].split(separator)
        #     triples = []
        #     for line in lines[1:]:
        #         if not line:
        #             continue
        #         values = line.split(separator)
        #         for i, v in enumerate(values):
        #             if i == 0:
        #                 entity = v
        #             else:
        #                 triples.append((entity, header[i], v))
        #     return triples

        def csv2triples(csv, separator='\t', delimiter='\n'):  #这里可以对于仿真数据集，可以按照规则排列一下三元组中前两个的顺序
            lines = csv.strip().split(delimiter)
            header = lines[0].split(separator)
            triples = []
            for line in lines[1:]:
                if not line:
                    continue
                values = line.split(separator)
                entity = values[0]
                for i in range(1, len(values)):
                    if i >= len(header):
                        break
                    triples.append((entity, header[i], values[i]))
            return triples

        def process_triplets(triplets):
            new_triplets = []
            for triplet in triplets:
                new_triplet = []
                triplet_temp = []
                if len(triplet) > 2:
                    if is_int(triplet[2]) or is_float(triplet[2]):
                        triplet_temp = (triplet[0].lower(), triplet[1].lower(), float(triplet[2]))
                    else:
                        triplet_temp = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
                else: 
                    triplet_temp = (triplet[0].lower(), triplet[1].lower(), "no meaning")
                new_triplets.append(triplet_temp)
            return new_triplets

        def intersection_with_tolerance(a, b, tol_word, tol_num):
            a = set(a)
            b = set(b)
            c = set()
            for elem1 in a:
                for elem2 in b:
                    if is_float(elem1[-1]) and is_float(elem2[-1]):
                        if (Levenshtein.distance(''.join(elem1[:-1]),''.join(elem2[:-1])) <= tol_word) and (abs(elem1[-1] - elem2[-1]) / (elem2[-1]+0.000001) <= tol_num):
                            c.add(elem1)
                    else:
                        if (Levenshtein.distance(''.join([str(i) for i in elem1]),''.join([str(j) for j in elem2])) <= tol_word):
                            c.add(elem1)
            return list(c)

        def union_with_tolerance(a, b, tol_word, tol_num):
            c = set(a) | set(b)
            d = set(a) & set(b)
            e = intersection_with_tolerance(a, b, tol_word, tol_num)
            f = set(e)
            g = c-(f-d)
            return list(g)

        def get_eval_list(pred_csv, label_csv, separator='\\t', delimiter='\\n', tol_word=3, tol_num=0.05):
            pred_triple_list=[]
            for it in pred_csv:
                pred_triple_temp = csv2triples(it, separator=separator, delimiter=delimiter)
                pred_triple_pre = process_triplets(pred_triple_temp)
                pred_triple_list.append(pred_triple_pre) 

            label_triple_list=[]
            for it in label_csv:
                label_triple_temp = csv2triples(it, separator=separator, delimiter=delimiter)
                label_triple_pre = process_triplets(label_triple_temp)
                label_triple_list.append(label_triple_pre) 

            intersection_list=[]
            union_list=[]
            sim_list=[]
            for pred,label in zip(pred_triple_list, label_triple_list):
                intersection = intersection_with_tolerance(pred, label, tol_word = tol_word, tol_num=tol_num)
                union = union_with_tolerance(pred, label, tol_word = tol_word, tol_num=tol_num)
                sim = len(intersection)/len(union)
                intersection_list.append(intersection)
                union_list.append(union)
                sim_list.append(sim)
            return intersection_list, union_list, sim_list

        def get_ap(predictions, labels, sim_threhold, tolerance, separator='\\t', delimiter='\\n'):
            if tolerance == 'strict':
                tol_word=0
                tol_num=0
            elif tolerance == 'slight':
                tol_word=2
                tol_num=0.05
            elif tolerance == 'high':
                tol_word= 5
                tol_num=0.1       
            intersection_list, union_list, sim_list = get_eval_list(predictions, labels, separator=separator, delimiter=delimiter, tol_word=tol_word, tol_num=tol_num)
            ap = len([num for num in sim_list if num >= sim_threhold])/len(sim_list)
            return ap   

        map_strict = 0
        map_slight = 0
        map_high = 0

        for sim_threhold in np.arange (0.5, 1, 0.05):
            map_temp_strict = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='strict', separator='\\t', delimiter='\\n')
            map_temp_slight = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='slight', separator='\\t', delimiter='\\n')
            map_temp_high = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='high', separator='\\t', delimiter='\\n')
            map_strict += map_temp_strict/10
            map_slight += map_temp_slight/10
            map_high += map_temp_high/10

        em = get_ap(predictions, labels, sim_threhold=1, tolerance='strict', separator='\\t', delimiter='\\n')
        ap_50_strict = get_ap(predictions, labels, sim_threhold=0.5, tolerance='strict', separator='\\t', delimiter='\\n')
        ap_75_strict = get_ap(predictions, labels, sim_threhold=0.75, tolerance='strict', separator='\\t', delimiter='\\n')    
        ap_90_strict = get_ap(predictions, labels, sim_threhold=0.90, tolerance='strict', separator='\\t', delimiter='\\n')
        ap_50_slight = get_ap(predictions, labels, sim_threhold=0.5, tolerance='slight', separator='\\t', delimiter='\\n')
        ap_75_slight = get_ap(predictions, labels, sim_threhold=0.75, tolerance='slight', separator='\\t', delimiter='\\n')    
        ap_90_slight = get_ap(predictions, labels, sim_threhold=0.90, tolerance='slight', separator='\\t', delimiter='\\n')
        ap_50_high = get_ap(predictions, labels, sim_threhold=0.5, tolerance='high', separator='\\t', delimiter='\\n')
        ap_75_high = get_ap(predictions, labels, sim_threhold=0.75, tolerance='high', separator='\\t', delimiter='\\n')    
        ap_90_high = get_ap(predictions, labels, sim_threhold=0.90, tolerance='high', separator='\\t', delimiter='\\n')
        return em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high