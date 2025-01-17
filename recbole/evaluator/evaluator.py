# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict
from recbole.evaluator import TopkMetric


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        return result_dict

    def evaluate_user(self, mode='best'):
        """calculate the top users. It is called at the end of the entire training and evaluation

        Args:
            mode (str): 'best' if return best users, 'worst' to return worst evaluated users

        Returns:
            collections.OrderedDict: such as
                            ``{
                                'precision@10': [{'526': '0.5'}, {'477': '0.5'}],
                                'ndcg@10': [{'410': '0.6086'}, {'453': '0.5474'}]
                            }``
            consisting out of a list of (userIdx, score) pairs per metric
        """
        best_result_dict = OrderedDict()
        worst_result_dict = OrderedDict()
        for metric in self.metrics:
            if issubclass(self.metric_class[metric].__class__, TopkMetric):
                best_user_val = self.metric_class[metric].top_user_dict
                worst_user_val = self.metric_class[metric].worst_user_dict

                best_result_dict.update(best_user_val)
                worst_result_dict.update(worst_user_val)

        if mode == 'best':
            result_dict = best_result_dict
        elif mode == 'worst':
            result_dict = worst_result_dict
        else:
           raise NotImplementedError('Make sure "mode" for user evaluation is in ["best","worst"]')

        return result_dict
