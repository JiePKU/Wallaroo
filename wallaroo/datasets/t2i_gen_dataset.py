import itertools
import json
import math
import os
import random
import re
import glob
import torch
import copy
import traceback
import collections
import numpy as np
import pickle 
import torchvision.transforms as T

from io import BytesIO
from PIL import Image
from functools import partial
from typing import List, Optional, Union

from .lib.dataloader.dataset_mem import DatasetMem
from data_curation import TFReader
from torch.utils.data import default_collate
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BertTokenizer, AutoProcessor, Qwen2VLImageProcessor
from wallaroo.utils.dist_utils import get_world_size, clip_grad_norm_, get_local_rank, get_rank
from .constants import *
from .utils import llava_to_openai, field_probabilities

class Text2ImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, resolution, processor, max_seq_length, vqvae_downscale=16, hw_indicator=False, t2i_generate_method='line_break', **kwargs):
        self.root = root
        self.transform = T.Compose([
                                        T.Resize(resolution, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                                        T.CenterCrop(resolution),
                                        T.ToTensor(),
                                        T.Normalize([.5], [.5]),
                                    ])
        self.resolution = resolution
        self.uncond_p = kwargs.get('uncond_p', 0.1)
        print('t2i uncond_p: ', self.uncond_p)
        self.cn_prompt_sample_prob = kwargs.get('cn_prompt_sample_prob', 0.5)
        self.processor = processor
        self.vqvae_downscale = vqvae_downscale
        self.max_seq_length = max_seq_length
        self.t2i_generate_method = t2i_generate_method
        self.hw_indicator = hw_indicator

        # generate dataset
        data_paths = root.split(":")
        print("t2i all data paths: ", data_paths)
        self.data_files = []
        for data_path in data_paths:
            data_file = glob.glob(os.path.join(data_path, "*.tfrecord"))
            self.data_files.extend(data_file)
        
        random.shuffle(self.data_files)

        # read tfrecord
        self.desc_dict = {'state_shape': 'byte', 'state_dict': 'byte', 'meta': 'byte'}  
        self.ori_imgs_nums = -1

    
    def clean(self, prompt):
        if prompt is None or prompt == '':
            return None
        tmp = prompt.split('**')
        if tmp[0] == '':
            tmp = tmp[1:]

        if len(tmp) == 0:
            return None
        else:
            tmp = tmp[0]

        tmp = tmp.split('--')[0].strip()
        return tmp

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        world_size = get_world_size()
        rank = get_rank()
        self.start = 0
        self.end = len(self.data_files)
        if worker_info is None:  
            iter_start = self.start
            iter_end = self.end
        else:  
            per_worker = (self.end - self.start) // (worker_info.num_workers * world_size)
            worker_id = worker_info.id
            iter_start = self.start + (worker_id + rank * worker_info.num_workers) * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        self.sub_data_files = self.data_files[iter_start: iter_end]
        random.seed(rank * worker_info.num_workers + worker_info.id)
        print(f"rank_{rank}:iter_start_{iter_start}:iter_end_{iter_end}:total_{len(self.data_files)}")

        data_files = copy.deepcopy(self.data_files)
        random.shuffle(data_files)
        print("before", len(self.sub_data_files))
        self.sub_data_files = self.sub_data_files + data_files
        print("after", len(self.sub_data_files))
        for file_name in self.sub_data_files:
            print(f"rank {rank}, reading_file is {file_name}")
            reader = TFReader(file_name, return_as_data_meta=False)
            iter_reader = iter(reader)
            while True:
                try:
                    meta, raw_dict = next(iter_reader)
                    meta = json.loads(meta)

                    data_source = meta.get('variables').get('data_source', None)
                    assert data_source in ['blip3o_60k', 'Share4o', 'OpenGPT-4o']


                    ori_w, ori_h = meta.get('variables').get('img_h'), meta.get('variables').get('img_w')
                        
                    aspect_ratio = ori_h / ori_w

                    if max(ori_w, ori_h) < 256 or aspect_ratio < 0.5 or aspect_ratio > 2:
                        continue

                    if data_source in ['blip3o_60k', 'Share4o', 'OpenGPT-4o']:
                        prompt_field = []
                        if 'prompt_en' in meta.get('variables') and isinstance(meta.get('variables').get('prompt_en'), str):
                            prompt_field.append('prompt_en')
                        if 'conversations' in meta.get('variables') and isinstance(meta.get('variables').get('conversations'), str):
                            prompt_field.append('conversations')
                        
                        sel_prompt = random.choice(prompt_field)
                        clean_prompt = meta.get('variables').get(sel_prompt)
                    
                    if clean_prompt.startswith('"') and clean_prompt.endswith('"'):
                        clean_prompt = clean_prompt.strip('"')


                    # print(data_source, clean_prompt)
                    if random.random() < self.uncond_p:
                        clean_prompt = ''
                    
                    buffer = BytesIO(raw_dict['raw_image'])
                    im_pil = Image.open(buffer)
                    if im_pil.mode != 'RGB':
                        im_pil = im_pil.convert('RGB')
                
                    image = self.transform(im_pil)
                    tgt_h, tgt_w =  self.resolution, self.resolution

                    generate_img_grid_thw = torch.tensor([1, tgt_h // (self.vqvae_downscale // 2), tgt_w // (self.vqvae_downscale // 2)])

                    vq_num_tokens = image.shape[-1] * image.shape[-2] // (self.vqvae_downscale * self.vqvae_downscale)

                    sources = {}
                    conversations = []

                    rsp_img_tokens =  GENERATE_START_TOKEN + f"<indicator:{tgt_h}>" + f"<indicator:{tgt_w}>" if self.hw_indicator else GENERATE_START_TOKEN
                    if self.t2i_generate_method == 'line_break':
                        for i in range(tgt_h // self.vqvae_downscale):
                            rsp_img_tokens += DEFAULT_GENERATE_IMAGE_TOKEN * (tgt_w // self.vqvae_downscale)
                            rsp_img_tokens = rsp_img_tokens + EOL_TOKEN if  i != tgt_h // self.vqvae_downscale - 1 else rsp_img_tokens + GENERATE_END_TOKEN
                    elif self.t2i_generate_method == 'no_line':
                        for i in range(tgt_h // self.vqvae_downscale):
                            rsp_img_tokens += DEFAULT_GENERATE_IMAGE_TOKEN * (tgt_w // self.vqvae_downscale)
                        rsp_img_tokens += GENERATE_END_TOKEN
                    elif self.t2i_generate_method == 'diffusion': # no_line
                        for i in range(tgt_h // self.vqvae_downscale):
                            rsp_img_tokens += DEFAULT_GENERATE_IMAGE_TOKEN * (tgt_w // self.vqvae_downscale)
                        rsp_img_tokens + GENERATE_END_TOKEN
                    else:
                        raise f"only support t2i generate method in [line_break], but input method is {self.t2i_generate_method}"

                    if 'cn' in sel_prompt:
                        conversations.append({'from': 'human', 'value': random.choice(chinese_generate_template).format(target_w=tgt_w, target_h=tgt_h, prompt=clean_prompt)}) 
                    else:
                        conversations.append({'from': 'human', 'value': random.choice(english_generate_template).format(target_w=tgt_w, target_h=tgt_h, prompt=clean_prompt)}) 
                    conversations.append({'from': 'gpt', 'value': rsp_img_tokens})
                    
                    sources['conversations'] = conversations
                    sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=False))

                    all_input_ids = [] 
                    all_labels = []

                    if len(SYSTEM_MESSAGE) > 0:
                        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
                        system_message_input_ids = self.processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
                        system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
                        
                        all_input_ids.append(system_message_input_ids.squeeze(0))
                        all_labels.append(system_labels.squeeze(0))

                    for _, j in enumerate(range(0, len(sources), 2)):
                        user_input = sources[j]
                        gpt_response = sources[j + 1]

                        user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                        gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
                        
                        prompt_input_ids = self.processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                        response_input_ids = self.processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                        labels = torch.cat(
                            [
                                torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                                response_input_ids.squeeze(0),
                            ],
                            dim=0,
                        )

                        all_input_ids.append(input_ids)
                        all_labels.append(labels)

                    # There is no need for eos or bos tokens in the input_ids
                    # Qwen2-VL does not use them
                    
                    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
                    labels = torch.cat(all_labels, dim=0).to(torch.long)
                    
                    if input_ids.shape[0] > self.max_seq_length:
                        continue

                    text_tokens_and_mask = self.processor.tokenizer.pad(
                                                            {'input_ids': [input_ids]},
                                                            max_length=self.max_seq_length,
                                                            padding='max_length',
                                                            # truncation=True,
                                                            return_attention_mask=True,
                                                            padding_side='right',
                                                            return_tensors='pt')

                    labels = torch.nn.functional.pad(labels, (0, self.max_seq_length - labels.shape[0]), value=-100)

                    data_dict = dict(
                        input_ids=text_tokens_and_mask['input_ids'].squeeze(0),
                        attention_mask=text_tokens_and_mask['attention_mask'].squeeze(0),
                        labels=labels,
                        images=image,
                        image_grid_thw=generate_img_grid_thw
                    )
                    yield data_dict

                except StopIteration:
                    break
                except Exception as e:
                    print(f"error in data loading: {e}")
                    traceback.print_exc()
                    continue
                    
    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('images'):
                batched[k] = torch.stack(v, dim=0)

        return batched