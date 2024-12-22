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
from recbole.data.dataset import Dataset
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

    def evaluate_user(self, dataset: Dataset):
        """calculate the top users. It is called at the end of the entire training and evaluation

        Args:
            dataset (Dataset): the valid data, default: None.

        Returns:
            collections.OrderedDict: such as
                            ``{
                                'precision@10': [{'526': '0.5'}, {'477': '0.5'}],
                                'ndcg@10': [{'410': '0.6086'}, {'453': '0.5474'}]
                            }``
            consisting out of a list of (userIdx, score) pairs per metric
        """
        result_dict = OrderedDict()
        for metric in self.metrics:
            if issubclass(self.metric_class[metric].__class__, TopkMetric):
                user_val = self.metric_class[metric].user_dict
                result_dict.update(user_val)
                print(dataset.inter_num)
        return result_dict
