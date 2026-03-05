# Copyright (c) Meituan. All rights reserved.
import concurrent.futures
import json
import os
import pickle
import random
import subprocess
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional, Union

import numpy as np
import psutil
from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset
from torch.utils.data import IterableDataset, get_worker_info

from data_curation.logger import logger
from data_curation.utils import (decode_image_with_cv2, get_index_filename, remove_ext)

cpu_count = psutil.cpu_count()


def items_in_tfindex(tfindex_filelist, accurate=False):
    if isinstance(tfindex_filelist, str):
        tfindex_filelist = [tfindex_filelist]

    if len(tfindex_filelist) > cpu_count and not accurate:
        logger.warning('An approximate number of items will be returned.')
        factor = len(tfindex_filelist) / cpu_count
        tfindex_filelist = random.choices(tfindex_filelist, k=cpu_count)
    else:
        factor = 1.0

    def func(tfindex):
        result = subprocess.run(['wc', '-l', tfindex], stdout=subprocess.PIPE)
        line_count = int(result.stdout.decode('utf-8').split()[0])
        return line_count

    items_in_total = 0
    with concurrent.futures.ThreadPoolExecutor(cpu_count) as executor:
        buffer = [
            executor.submit(func, tfindex) for tfindex in tfindex_filelist
        ]
        for task in concurrent.futures.as_completed(buffer):
            items_in_total += task.result()

    return int(items_in_total * factor)


# Function to decode data
def decode_data(data):
    """Function to decode data.

    Args:
        data: The data to be decoded.
    Returns:
        tuple: The decoded metadata and state dictionary.
    """
    meta_str = data['meta'].decode()

    state_shape = json.loads(data['state_shape'].decode())

    # 创建一个要删除的键的列表
    keys_to_delete = [k for k in state_shape.keys() if 'embedding' in k]

    raw_dict = pickle.loads(data['state_dict'])
    
    state_dict = {}
    for k, v in state_shape.items():
        if k in keys_to_delete:
            continue
        if k in ['image', 'frames']:
            ndarray = np.frombuffer(raw_dict[k], dtype=np.uint8)
            if k == 'image':
                image = decode_image_with_cv2(ndarray)
                state_dict[k] = image
            else:
                ndarrays = np.split(ndarray,
                                    indices_or_sections=np.cumsum(v)[:-1])
                frames = []
                for ndarray in ndarrays:
                    image = decode_image_with_cv2(ndarray)
                    frames.append(image)
                state_dict[k] = np.stack(frames, axis=0)
        elif k.startswith('raw_'):
            state_dict[k] = raw_dict[k]
        elif k == "flan_t5_xxl_embedding" or k.endswith("fp16"):
            state_dict[k] = np.frombuffer(raw_dict[k],
                                        dtype=np.float16).reshape(v)
        elif k == "flan_t5_xxl_mask" or k.endswith("bool"):
            state_dict[k] = np.frombuffer(raw_dict[k],
                                        dtype=bool).reshape(v)
        else:
            state_dict[k] = np.frombuffer(raw_dict[k], dtype=np.float32).reshape(v)

    return meta_str, state_dict


# Define TFReader class
class TFReader(TFRecordDataset):
    """How to use `TFReader` correctly?

    `TFReader` is used to read the specified `tfrecord` file. When `shuffle_queue_size` is not specified, it will sequentially return each
    entry in the file. We recommend using it for loading test data, or for quickly reading the contents of a file.
    """

    def __init__(self,
                 data_path,
                 index_path=None,
                 transform=None,
                 return_as_data_meta: bool = False,
                 load_raw_data: bool = False,
                 **kwargs):
        """Initialize the TFReader.

        Args:
            data_path: The path to the data.
            index_path: Optional; if None, will try to read data_path.tfindex or create it automatically.
            transform: Optional; the transformation to apply to the data.
            return_as_data_meta: If True, the data will be returned as a DataMeta object. If False, data will be returned as (meta, data).
            load_raw_data: If True and return_as_data_meta is also True, will call `data_meta.load_raw_data()`
                to load raw data before applying the `transform` function.
        """
        context_description = {
            c: 'byte'
            for c in ['meta', 'state_shape', 'state_dict']
        }

        def _transform(x):
            meta, data = decode_data(x)
            if return_as_data_meta:
                import data_curation  # noqa: F401
                from data_curation.data_meta import DataMeta
                dm = DataMeta.from_record(meta, data)
                if load_raw_data:
                    dm.load_raw_data()
                if transform is not None:
                    return transform(dm)
                else:
                    return dm
            else:
                if transform is not None:
                    return transform(meta, data)
                else:
                    return meta, data

        self.data_path = data_path
        if index_path is None:
            tmp = get_index_filename(data_path)
            if os.path.exists(tmp):
                index_path = tmp
                logger.debug(f'Set index_path as {index_path}')
        self.index_path = index_path

        super().__init__(data_path=data_path,
                         index_path=index_path,
                         transform=_transform,
                         description=context_description,
                         **kwargs)

    # Method to represent the TFReader
    def __repr__(self):
        """Represent the TFReader.

        Returns:
            str: The representation of the TFReader.
        """
        return f'{self.__class__.__name__}({self.data_path})'

    @property
    def approximate_length(self):
        if hasattr(self, '_approximate_length'):
            return getattr(self, '_approximate_length')
        assert self.index_path is not None, 'Index file is not specified.'
        total_items = items_in_tfindex(self.index_path)
        setattr(self, '_approximate_length', total_items)
        return total_items