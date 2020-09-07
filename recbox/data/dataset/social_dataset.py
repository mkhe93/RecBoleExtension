# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

import os
from logging import getLogger

import torch
import dgl

from .dataset import Dataset
from ...utils import FeatureSource


class SocialDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()

        if saved_dataset is None:
            self._from_scratch(config)
        else:
            self._restore_saved_dataset(saved_dataset)

    def _from_scratch(self, config):
        self.dataset_path = config['data_path']
        self._fill_nan_flag = self.config['fill_nan']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2seqlen = config['seq_len'] or {}

        self.model_type = self.config['MODEL_TYPE']
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        self.source_field = self.config['SOURCE_ID_FIELD']
        self.target_field = self.config['TARGET_ID_FIELD']
        self._check_field('source_field', 'target_field')

        self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.dataset_name, self.dataset_path)
        self.feat_list = [feat for feat in [self.inter_feat, self.user_feat, self.item_feat, self.net_feat] if feat is not None]

        self._filter_by_inter_num()
        self._filter_by_field_value()
        self._reset_index()

        self.net_feat = self._load_net(self.dataset_name, self.dataset_path)

        self._remap_ID_all()

        self._user_item_feat_preparation()

        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()
        
        self.dgl_graph = self.create_dgl_graph()

    def _load_net(self, dataset_name, dataset_path): 
        net_file_path = os.path.join(dataset_path, '{}.{}'.format(dataset_name, 'net'))
        if os.path.isfile(net_file_path):
            return self._load_feat(net_file_path, FeatureSource.NET)
        else:
            raise ValueError('File {} not exist'.format(net_file_path))
            
    def _get_fields_in_same_space(self):
        fields_in_same_space = super()._get_fields_in_same_space()
        fields_in_same_space = [_ for _ in fields_in_same_space if (self.source_field not in _) and
                                                                   (self.target_field not in _)]
        for field_set in fields_in_same_space:
            if self.uid_field in field_set:
                field_set.update({self.source_field, self.target_field})

        return fields_in_same_space

    def create_dgl_graph(self):
        if self.net_feat is not None:
            source = torch.tensor(self.net_feat[self.source_field].values)
            target = torch.tensor(self.net_feat[self.target_field].values)
            ret = dgl.graph((source, target))

            # TODO add edge feature

            return ret
        else:
            raise ValueError('net_feat does not exist')

    def __str__(self):
        info = []
        if self.uid_field:
            info.extend(['The number of users: {}'.format(self.user_num),
                         'Average actions of users: {}'.format(self.avg_actions_of_users)])
        if self.iid_field:
            info.extend(['The number of items: {}'.format(self.item_num),
                         'Average actions of items: {}'.format(self.avg_actions_of_items)])
        if self.dgl_graph is not None:
            info.extend(['The number of nodes: {}'.format(self.dgl_graph.number_of_nodes()),
                         'The number of edges: {}'.format(self.dgl_graph.number_of_edges())])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            info.append('The sparsity of the dataset: {}%'.format(self.sparsity * 100))

        info.append('Remain Fields: {}'.format(list(self.field2type)))
        return '\n'.join(info)