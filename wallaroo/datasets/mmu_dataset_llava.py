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
import torchvision.transforms as T

from io import BytesIO
from PIL import Image
from functools import partial
from typing import List, Optional, Union

from data_curation import TFReader
from torch.utils.data import default_collate
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BertTokenizer
from wallaroo.utils.dist_utils import get_world_size, get_rank
from .constants import *
from .utils import llava_to_openai,  get_media_info


def check_image_token(conversations):
    for conversation in conversations:
        if conversation['from'] == 'human':
            if '<image>' in conversation['value']:
                return True
    
    return False
    
def replace_image_wth_video(conversations):

    for conversation in conversations:
        if conversation['from'] == 'human':
            if '<image>' in conversation['value']:
                conversation['value'] = conversation['value'].replace('<image>', '<video>')

    return conversations


def image_token_check(input_ids, image_thw):
    mask = input_ids==151655
    nums = mask.sum().item()
    size = image_thw.shape[0]
    for i in range(size):
        h, w = image_thw[i][1], image_thw[i][2]
        assert h %2 ==0 and w%2==0
        nums -= h//2 * w//2
    if nums==0:
        return True
    return False


def video_token_check(input_ids, video_thw):
    mask = input_ids==151656
    nums = mask.sum().item()
    size = video_thw.shape[0]
    for i in range(size):
        t, h, w = video_thw[i][0], video_thw[i][1], video_thw[i][2]
        assert h %2 ==0 and w%2==0
        nums -= h//2 * w//2 * t
    if nums==0:
        return True
    return False


class LLAVAMmuDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, processor, max_seq_length, data_args, **kwargs):
        self.root = root

        self.processor = processor
        self.max_seq_length = max_seq_length

        self.semantic_start_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.semantic_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

        self.image_min_pixel = data_args['seg_image_min_pixels']
        self.image_max_pixel = data_args['seg_image_max_pixels']
        self.size_factor = data_args['seg_size_factor']
        self.max_ratio = data_args['max_ratio']

        # generate dataset
        data_paths = root.split(":")
        print("mmu all data paths: ", data_paths)
        self.data_files = []
        for data_path in data_paths:
            data_file = glob.glob(os.path.join(data_path, "*.tfrecord"))
            self.data_files.extend(data_file)
        
        random.shuffle(self.data_files)

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
                    is_video = False
                    
                    data_rescource = meta['variables']['data_source']

                    if  data_rescource not in ['M4-Instruction', 'LLAVA-NeXT-Data', 'LLaVA-Video-178K', 'LLAVA-OneVision', 'LLaVA-CC3M-Pretrain', 'llava-en-zh-300k', \
                                                'llava-critic-113k', 'clevr_count_70k', 'llava-med-zh-instruct',  'multimodal-vqa-self-instruct-enriched', 'MMPR', \
                                                'food-visual-instructions', 'videochat2', 'Llama-Nemotron-VLM', 'RICO-ScreenQA', 'GQA', 'iconqa', 'Infinity_Instruct_7M_core', 'commonsense_qa', 'sft_bench', 'MAmmoTH-VL-Instruct-12M', 'mmu_v2', 'gpt4o', 'LLaVA-OneVision-1.5']:  # for internel mmu data
                        meta['variables']['without_image'] = 'no'
                        
                    if meta['variables']['without_image'] == 'no':  ## has image/video in this data record

                        meta_kyes = meta['variables'].keys() 
                        
                        ## "video" -> data path  "multi image" is a [data path] list 

                        if "image_path" in meta_kyes: ## multi image record 

                            image_inputs = get_media_info(meta['variables']["image_path"], self.image_min_pixel, self.image_max_pixel, is_video=False)

                        elif "video_path" in meta_kyes: ## video 
                            
                            is_video = True
                            video_inputs, video_kwargs = get_media_info([meta['variables']["video_path"]], self.image_min_pixel, self.image_max_pixel, is_video=True)

                        else: # single image
                            
                            buffer = BytesIO(raw_dict['raw_image'])
                            image = Image.open(buffer)

                            image_inputs = get_media_info([image], self.image_min_pixel, self.image_max_pixel, is_video=False)

                    
                    sources = {}
                    sources['conversations'] = meta['variables']['conversations'] 

                    if  data_rescource not in ['M4-Instruction', 'LLAVA-NeXT-Data', 'LLaVA-Video-178K', 'LLAVA-OneVision', 'LLaVA-CC3M-Pretrain', 'llava-en-zh-300k', \
                                                'llava-critic-113k', 'clevr_count_70k', 'llava-med-zh-instruct',  'multimodal-vqa-self-instruct-enriched', 'MMPR', \
                                                'food-visual-instructions', 'videochat2', 'Llama-Nemotron-VLM', 'RICO-ScreenQA', 'GQA', 'iconqa', 'Infinity_Instruct_7M_core', 'commonsense_qa', 'sft_bench', 'MAmmoTH-VL-Instruct-12M', 'mmu_v2', 'gpt4o', 'LLaVA-OneVision-1.5']: # for internel mmu data
                        conversations = []
                        conversations_list = meta['variables']['conversations']

                        try:
                            for _, j in enumerate(range(0, len(conversations_list), 2)):
                                conversations.append({'from': 'human', 'value': '<image>' + conversations_list[j] if j == 0 else conversations_list[j]}) 
                                conversations.append({'from': 'gpt', 'value': conversations_list[j+1]})
                        except:
                            continue

                        sources['conversations'] = conversations

                    ## We consider the case where <image> token is missing in all conversations while image is provided
                    if meta['variables']['without_image'] == 'no' and is_video==False and check_image_token(sources['conversations'])==False:
                        ## add '<image>' to the first conversation
                        sources['conversations'][0]['value'] = '<image>\n'+sources['conversations'][0]['value']
                    
                    if is_video: # video flag
                        ## replace <image> with <video> 
                        sources['conversations'] = replace_image_wth_video(sources['conversations'])
                        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=True))
                    else:
                        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=False))


                    all_input_ids = [] 
                    all_labels = []
                    all_semantic_values = []
                    all_image_grid_thw = []
                    all_video_grid_thw = []
                    all_second_grid = []

                    # Qwen2.5-VL uses a default system message so I've added this.
                    if len(SYSTEM_MESSAGE) > 0:
                        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
                        system_message_input_ids = self.processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
                        system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
                        
                        all_input_ids.append(system_message_input_ids.squeeze(0))
                        all_labels.append(system_labels.squeeze(0))

                    
                    start_index = 0

                    for _, j in enumerate(range(0, len(sources), 2)):

                        user_input = sources[j]
                        gpt_response = sources[j + 1]
                        
                        user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                        gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

                        ## deal with (multi) imgae
                        if DEFAULT_IMAGE_TOKEN in user_input:
                            
                            ## compute the number of DEFAULT_IMAGE_TOKEN in user_inputs
                            nums = user_input.count(DEFAULT_IMAGE_TOKEN)       
                                                 
                            inputs = self.processor(text=[user_input], images=image_inputs[start_index:start_index+nums], videos=None, padding=False, return_tensors='pt')
                            
                            prompt_input_ids = inputs['input_ids']
                            all_semantic_values.append(inputs['pixel_values'])
                            all_image_grid_thw.append(inputs["image_grid_thw"])
                            
                            start_index += nums 

                        ## deal with video
                        elif DEFAULT_VIDEO_TOKEN in user_input:
                            
                            nums = user_input.count(DEFAULT_VIDEO_TOKEN)

                            inputs = self.processor(text=[user_input], images=None, videos=video_inputs[start_index:start_index+nums], padding=False, return_tensors='pt', **video_kwargs)
                            
                            
                            prompt_input_ids = inputs['input_ids']
                            all_video_grid_thw.append(inputs["video_grid_thw"])
                            all_semantic_values.append(inputs['pixel_values_videos'])
                            all_second_grid.append(inputs['second_per_grid_ts'][0])

                            start_index += nums

                        ## deal with pure language
                        else:
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

                    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
                    labels = torch.cat(all_labels, dim=0).to(torch.long)

                    # print("mmu sample length", input_ids.shape[0])
                    if input_ids.shape[0] > self.max_seq_length:
                        print("mmu pass max length", input_ids.shape[0], data_rescource)
                        continue

                    text_tokens_and_mask = self.processor.tokenizer.pad(
                                                                        {'input_ids': [input_ids]},
                                                                        max_length=self.max_seq_length,
                                                                        padding='max_length',
                                                                        return_attention_mask=True,
                                                                        padding_side='right',
                                                                        return_tensors='pt')

                    labels = torch.nn.functional.pad(labels, (0, self.max_seq_length - labels.shape[0]), value=-100)

                    data_dict = dict(
                        input_ids=text_tokens_and_mask['input_ids'].squeeze(0),
                        attention_mask=text_tokens_and_mask['attention_mask'].squeeze(0),
                        labels=labels,
                    )

                    ## when contain image and video
                    if len(all_semantic_values)>0:
                        
                        if is_video == False:
                            image_thw = torch.cat(all_image_grid_thw, dim=0)
                            data_dict['image_grid_thw'] = image_thw
                            data_dict['pixel_values'] = torch.cat(all_semantic_values , dim=0)

                            if image_token_check(text_tokens_and_mask['input_ids'].squeeze(0), image_thw)==False:
                                print("sequence fails to align in file", file_name, "    ******    ", meta['variables']['conversations'])
                                continue

                        else:
                            video_thw = torch.cat(all_video_grid_thw, dim=0)
                            data_dict['video_grid_thw'] = video_thw
                            data_dict['second_per_grid_ts'] = torch.tensor(all_second_grid)
                            data_dict['pixel_values_videos'] = torch.cat(all_semantic_values, dim=0)

                            if video_token_check(text_tokens_and_mask['input_ids'].squeeze(0), video_thw)==False:
                                print("sequence fails to align in file", file_name, "    ******    ", meta['variables']['conversations'])
                                continue
                            
                    yield data_dict

                except StopIteration:
                    break
                except Exception as e:
                    print(f"error in data loading: {e}, {file_name}")
                    traceback.print_exc()
                    continue
        
    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ("pixel_values", "pixel_values_videos", "image_grid_thw"):
                batched[k] = torch.stack(v, dim=0)

        return batched