import json
import os.path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class Node:
    def __init__(self, label_id=-1, level=-1, label=None, parent=-1):
        self.id = label_id
        self.label = label
        self.level = level
        self.child = []
        self.parent = -1

    def set_child(self, child):
        self.child.append(child)


def get_label(path):
    with open(path, "r") as f:
        label_id_level = json.load(f)
    return label_id_level


class MyDataSet:
    def __init__(self, path):
        self.train_path = os.path.join(path, "train.pkl")
        self.valid_path = os.path.join(path, "valid.pkl")
        self.test_path = os.path.join(path, "test.pkl")
        self.tree = Node(-1, 0, 'Root')
        self.tree_map = {}
        self.id2label = {}
        self.label2id = {}
        self.parent = {}
        self.label_nums = -1
        self.build_tree(os.path.join(path, "label.taxonomy"))
        # create label list level by level
        self.level_label, self.id2level, self.id2child = self.get_level_label()
        self.path_mode = False

    def build_tree(self, path):
        label_id = 0
        with open(path, 'r') as f:
            line = f.readline()
            i = 0
            while line:
                labels = line.split('\n')[0].split('\t')
                if i == 0:
                    self.tree_map['Root'] = self.tree
                    for label in labels[1:]:
                        temp = Node(label_id, self.tree.level + 1, label, -1)
                        self.parent[label_id] = -1
                        self.tree.set_child(temp)
                        self.tree_map[label] = temp
                        self.label2id[label] = label_id
                        self.id2label[label_id] = label
                        label_id += 1
                        i += 1
                else:
                    root_temp = self.tree_map[labels[0]]
                    for label in labels[1:]:
                        temp = Node(label_id, root_temp.level +
                                    1, label, root_temp.id)
                        self.parent[label_id] = root_temp.id
                        root_temp.set_child(temp)
                        self.tree_map[label] = temp
                        self.label2id[label] = label_id
                        self.id2label[label_id] = label
                        label_id += 1
                line = f.readline()
        self.label_nums = label_id

    def get_dfs_path(self, labels):
        result = []
        for node in self.tree.child:
            if node.label in labels:
                result.append(node.label)
                self.backtrack(labels, node, result)
        return result if len(result) != 0 else []

    def backtrack(self, labels, root, result):
        if len(root.child) == 0:
            return
        else:
            for node in root.child:
                if node.label in labels:
                    result.append(node.label)
                    self.backtrack(labels, node, result)

    # def backtrack(self, labels, temp, root, result):
    #     if len(root.child) == 0:
    #         for label in temp:
    #             result.append(label)
    #         result.append('&')
    #         return
    #     else:
    #         flag = 0
    #         for node in root.child:
    #             if node.label in labels:
    #                 flag = 1
    #                 temp.append(node.label)
    #                 self.backtrack(labels, temp, node, result)
    #                 temp.pop()
    #         if flag == 0:
    #             for label in temp:
    #                 result.append(label)
    #             result.append('&')
    #             return

    def get_train_data(self):
        return read_data(self.train_path)

    def get_valid_data(self):
        return read_data(self.valid_path)

    def get_test_data(self):
        return read_data(self.test_path)

    def get_path_mode_data(self, data):
        # add '&'

        new_data = []
        for d in data:
            all_path = self.get_dfs_path(d[1])
            if len(set(all_path)) != len(d[1]) + 1 and '&' in all_path:
                print(all_path)
                print(d[1])
            new_data.append((d[0], all_path, d[2]))
        return new_data

    def append_item(self):
        # self.label_id_level['&'] = {'id': self.label_nums, 'level': -1}
        self.label2id['&'] = self.label_nums
        self.id2label[self.label_nums] = '&'
        # self.label_nums += 1

    def get_labels_vector(self, labels):
        labels_vector = []
        for i in range(len(labels)):
            temp = [0] * self.label_nums
            for label in labels[i]:
                temp[self.label2id[label]] = 1
            labels_vector.append(temp)
        return labels_vector

    def get_label_order_data(self, data):
        new_data = []
        for d in data:
            temp = []
            for i in range(len(d[2])):
                if d[2][i] == 1:
                    temp.append(self.id2label[i])
            new_data.append((d[0], temp, d[2]))
        return new_data

    def get_data(self, mode, data_type):
        if data_type == 'train':
            data = self.get_train_data()
        elif data_type == 'valid':
            data = self.get_valid_data()
        elif data_type == 'test':
            data = self.get_test_data()
        else:
            raise RuntimeError(
                "No data type was selected from ['train', 'valid', 'test']")
        labels = [d[1] for d in data]
        labels_vector = self.get_labels_vector(labels)
        new_data = [(data[i][0], data[i][1], labels_vector[i])
                    for i in range(len(data))]
        if mode == 'bfs':
            return self.get_label_order_data(new_data)
        elif mode == 'random_each_epoch':
            return new_data
        elif mode == 'random':
            return get_random_label(new_data)
        elif mode == 'dfs':
            if not self.path_mode:
                self.path_mode = True
                self.append_item()
            return self.get_path_mode_data(new_data)
        else:
            raise RuntimeError(
                "No mode was selected from [random, bfs, dfs, random_each_epoch]")

    def get_level_label(self):
        level_label = {}
        id2level = {}
        id2child = {}
        max_level = -1
        for value in self.tree_map.values():
            if max_level < value.level:
                max_level = value.level
        level_label['max_level'] = max_level
        for i in range(max_level):
            level_label[i + 1] = []
        for value in self.tree_map.values():
            if value.id != -1:
                level_label[value.level].append(value.id)
                id2level[value.id] = value.level
                temp = []
                id2child[value.id] = []
                for v in value.child:
                    id2child[value.id].append(v.id)
        id2child[-1] = []
        for v in self.tree_map['Root'].child:
            id2child[-1].append(v.id)
        return level_label, id2level, id2child


def read_data(path):
    # 获取数据 (text, label)
    if 'pkl' in str(path):
        with open(str(path), 'rb') as f:
            data = pickle.load(f)
    return data


def get_random_label(data):
    # 获取随机标签数据
    for d in data:
        np.random.shuffle(d[1])
    return data


def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


class MyDataloader:
    def __init__(self, config, batch_size, dataset):
        self.label2id = dataset.label2id
        self.class_num = dataset.label_nums
        self.mode = config['mode']
        self.batch_size = batch_size
        self.reversed = config['reversed']
        self.dataset = dataset
        self.negative_sample = config['negative_sample']
        self.label_max_length = config[config['dataset']]['label_max_length']
        self.text_max_length = config['text_max_length']
        self.random_negative_sample = config['random_negative_sample']
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['pretrained_path'])

    def get_dataloader(self, data):
        # sampler = SequentialSampler(data)
        dataloader = DataLoader(
            data, batch_size=self.batch_size, collate_fn=self.get_batch)
        return dataloader

    def get_batch(self, data):
        """
        define: start = 0, end = 2, pad = 1
        """
        texts = [d[0] for d in data]
        labels = [d[1] for d in data]
        target = [d[2] for d in data]
        dec, dec_mask, labels = self.get_label_features(labels)
        # negative_sample = np.ones_like(labels)
        negative_sample = None
        if self.negative_sample:
            negative_sample = self.get_negative_sample(labels)
        elif self.random_negative_sample:
            negative_sample = self.get_random_negative_sample(labels)

        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt', max_length=self.text_max_length)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        # input_len = torch.sum(attention_mask, dim=1)
        dec = torch.tensor(dec, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        dec_mask = torch.tensor(dec_mask, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        if negative_sample is not None:
            negative_sample = torch.tensor(negative_sample, dtype=torch.long)

        return input_ids, attention_mask, dec, dec_mask, labels, target, negative_sample

    def get_label_features(self, labels):

        dec = []
        dec_mask = []
        target = []
        for i in range(len(labels)):
            # [start, x1, x2,...,xn]
            temp_dec = [0]
            temp_dec_mask = [1]
            # [x1, x2, ..., xn, end]
            temp_target = [2]
            if self.mode == 'random_each_epoch':
                np.random.shuffle(labels[i])
            if self.reversed:
                labels[i] = reversed(labels[i])
            for label in labels[i]:
                temp_dec_mask.append(1)
                temp_dec.append(self.label2id[label] + 3)
                temp_target.insert(-1, self.label2id[label] + 3)
            temp_len = len(temp_dec)
            while temp_len < self.label_max_length:
                temp_dec.append(1)
                temp_dec_mask.append(0)
                temp_target.append(1)
                temp_len += 1
            dec.append(temp_dec)
            dec_mask.append(temp_dec_mask)
            target.append(temp_target)
        return dec, dec_mask, target

    def get_negative_sample(self, target_labels):
        """
        labels: batch * maxlength
        """
        parent = self.dataset.parent  # dict
        child = self.dataset.id2child  # dict
        id2level = self.dataset.id2level  # dict
        level_label = self.dataset.level_label  # dict{list}

        temp = np.array(target_labels) - 3
        negative_sample = -2 * np.ones_like(temp)
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                if temp[i][j] < 0:
                    negative_sample[i][j] = -1
                    break
                # 替换成同父亲非目标兄弟
                sibling = child[parent[temp[i][j]]]
                not_target_sibling = []
                # 获取非目标兄弟节点
                for sib in sibling:
                    if sib not in temp[i]:
                        not_target_sibling.append(sib)
                if not_target_sibling:
                    np.random.shuffle(not_target_sibling)
                    negative_sample[i][j] = not_target_sibling[0]
                else:
                    # 替换同层非目标节点
                    level_sibling = level_label[id2level[temp[i][j]]]
                    # 获取同层非目标节点
                    for sib in level_sibling:
                        if sib not in temp[i]:
                            not_target_sibling.append(sib)
                    if not_target_sibling:
                        np.random.shuffle(not_target_sibling)
                        negative_sample[i][j] = not_target_sibling[0]
                    else:
                        # 同层没有目标节点,直接跳过
                        negative_sample[i][j] = -2
        negative_sample += 3
        return negative_sample

    def get_random_negative_sample(self, target_labels):
        class_id = [i for i in range(self.class_num)]
        temp = np.array(target_labels) - 3
        negative_sample = -2 * np.ones_like(temp)
        for i in range(len(temp)):
            not_target_id = []
            for c in class_id:
                if c not in temp[i]:
                    not_target_id.append(c)
            for j in range(len(temp[i])):
                if temp[i][j] < 0:
                    negative_sample[i][j] = -1
                    break
                if not_target_id:
                    np.random.shuffle(not_target_id)
                    negative_sample[i][j] = not_target_id[0]
                else:
                    negative_sample[i][j] = -1
        negative_sample += 3
        return negative_sample
