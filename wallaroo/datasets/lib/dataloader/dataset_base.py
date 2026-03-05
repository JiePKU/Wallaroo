import concurrent.futures
from abc import ABCMeta
from concurrent.futures import as_completed
from typing import List

# from automaxprocs import maxprocs

# from lichee import plugin
# from lichee.dataset.io_reader.io_reader_base import BaseIOReader
from ..io_reader.io_reader_base import BaseIOReader, TFRecordReader
from ..tfr_config import get_cfg


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, cfg, data_path_list: List[str], desc_dict, training=True):
        """
        Base dataset implementation

        :param cfg: dataset config
        :param data_path_list: dataset file path list
        :param desc_file: dataset description file
        :param training: training or not => shuffle or not
        """
        self.cfg = cfg
        self.data_path_list = data_path_list
        self.training = training
        # init description config
        self.description = desc_dict
        # 初始化数据
        self.tfrecord_data_file_list = self.try_convert_to_tfrecord()
       
        
        # init data index
        self.data_index_list = self.get_indexes()
        # init dataset length
        self.data_len = self.get_data_len()

    def get_indexes(self):
        max_workers = 8  # todo
        # reader_cls: BaseIOReader = plugin.get_plugin(plugin.PluginType.DATA_IO_READER, self.cfg.DATASET.FORMAT)
        reader_cls = TFRecordReader()

        data_index_list = [None] * len(self.data_path_list)
        if len(data_index_list) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                fs = {executor.submit(reader_cls.get_index, data_path, self.description): i for i, data_path in
                      enumerate(self.data_path_list)}
                for future in as_completed(fs):
                    data_index_list[fs[future]] = future.result()
        else:  # hack for single file
            data_index_list[0] = reader_cls.get_index(self.data_path_list[0], self.description)
        return data_index_list

    def get_nth_data_file(self, index):
        '''
        :param index: index of target item
        :return: item file index, and item start & end offset
        '''
        for i, data_index in enumerate(self.data_index_list):
            if index < len(data_index):
                break
            index -= len(data_index)
        start_offset = data_index[index]
        end_offset = data_index[index + 1] if index + 1 < len(data_index) else None
        return i, (start_offset, end_offset)

    def get_data_len(self):
        data_len = 0
        for data_index in self.data_index_list:
            data_len += len(data_index)
        return data_len

    def get_desc(self):
        return self.description

    def try_convert_to_tfrecord(self):
        tfrecord_data_file_list = []
        for data_path in self.data_path_list:
            # reader_cls: BaseIOReader = plugin.get_plugin(plugin.PluginType.DATA_IO_READER, self.cfg.DATASET.FORMAT)
            reader_cls = TFRecordReader
            tfrecord_data_file_list.append(reader_cls.convert_to_tfrecord(data_path, self.description))
        return tfrecord_data_file_list

    def __len__(self):
        return self.data_len

    def transform(self, row):
        """
        transform data with field parsers

        :param row: the raw feature map {key1: raw_feature1, key2:raw_feature2...}
        :return: parsed feature map {key1:feature1, key2:feature2...}
        """

        return row

    def collate(self, batch):
        """
        collate data in a batch

        :param batch: list of item feature map [item1, item2, item3 ...], item with {key1:feature1, key2:feature2}
        :return: batched feature map for model with format {key1: batch_feature1, key2: batch_feature2...}
        """
        # record = {}

        # todo no parser
        # for parser in self.parsers:
        #     collate_result = parser.collate(batch)
        #     if collate_result is not None:
        #         record.update(collate_result)
        record = default_collate(batch)

        return record
