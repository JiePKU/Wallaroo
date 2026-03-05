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
from wallaroo.utils.dist_utils import get_world_size, clip_grad_norm_, get_local_rank, get_rank
from .constants import *
from .utils import llava_to_openai, field_probabilities, center_crop_arr, get_image_info, smart_resize, get_closest_ratio

class MARImage2ImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, resolution, processor, max_seq_length, aspect_ratio_type, w_pixel_vae, pixel_pad_left, data_args, vqvae_downscale=16, hw_indicator=False, i2i_generate_method='line_break', **kwargs):
        self.root = root
        self.transform = T.Compose([
                                        T.Resize(resolution, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                                        T.CenterCrop(resolution),
                                        T.ToTensor(),
                                        T.Normalize([.5], [.5]),
                                    ])
        
        self.resolution = resolution
        self.uncond_p = kwargs.get('uncond_p', 0.1)
        print('edit uncond_p: ', self.uncond_p)
        self.cn_prompt_sample_prob = kwargs.get('cn_prompt_sample_prob', 0.5)
        self.processor = processor
        self.vqvae_downscale = vqvae_downscale
        self.max_seq_length = max_seq_length
        self.i2i_generate_method = i2i_generate_method
        self.hw_indicator = hw_indicator

        self.w_pixel_vae = w_pixel_vae
        self.pixel_pad_left = pixel_pad_left
        
        self.grid_key = "image_grid_thw"
        self.semantic_key = "semantic_values"
        self.pixel_key = "pixel_values"
        self.pixel_grid_key = "pixel_grid_thw"
        self.pil_image_key = 'pil_image'
        self.generate_pixel_grid_key = "generate_pixel_grid_thw"

        self.semantic_start_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.semantic_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

        self.image_min_pixel = data_args['seg_image_min_pixels']
        self.image_max_pixel = data_args['seg_image_max_pixels']
        self.size_factor = data_args['seg_size_factor']
        self.max_ratio = data_args['max_ratio']

        if self.w_pixel_vae:
            self.aspect_ratio = eval(aspect_ratio_type)

        # generate dataset
        data_paths = root.split(":")
        print("i2i edit all data paths: ", data_paths)
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
        if worker_info is None:  # single process load data
            iter_start = self.start
            iter_end = self.end
        else:   # multiple process load data
            # assert (self.end - self.start) % (world_size * worker_info.num_workers) == 0
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

                    source_im_pil = Image.open(BytesIO(raw_dict['raw_image'])).convert('RGB')
                    if 'raw_image_edited' in raw_dict:
                        target_im_pil = Image.open(BytesIO(raw_dict['raw_image_edited'])).convert('RGB')
                    elif 'raw_edited_image' in raw_dict:
                        target_im_pil = Image.open(BytesIO(raw_dict['raw_edited_image'])).convert('RGB')
                    
                    ori_w, ori_h = source_im_pil.size
                    aspect_ratio = ori_h / ori_w

                    prompt_field = []
                    if 'instruction' in meta['variables']:
                        prompt_field.append('instruction')
                    if 'instruction_long' in meta['variables']:
                        prompt_field.append('instruction_long')
                    if 'instruction_cn' in meta['variables']:
                        prompt_field.append('instruction_cn')
                    if 'instruction_en' in meta['variables']:
                        prompt_field.append('instruction_en')
                    if 'instruction_enrich' in meta['variables']:
                        prompt_field.append('instruction_enrich')
                    sel_prompt = random.choice(prompt_field)
                    clean_prompt = meta.get('variables').get(sel_prompt)
                    
                    if clean_prompt.startswith('"') and clean_prompt.endswith('"'):
                        clean_prompt = clean_prompt.strip('"')
                    
                    if random.random() < self.uncond_p:
                        clean_prompt = ''
                        source_im_pil = Image.new('RGB', source_im_pil.size, (0, 0, 0))
                    
                    
                    if '<image>' in clean_prompt:
                        continue

                    width, height = source_im_pil.size
                    seg_resized_height, seg_resized_width = smart_resize(height, width, factor=self.size_factor, min_pixels=self.image_min_pixel, max_pixels=self.image_max_pixel, max_ratio=self.max_ratio)

                    if self.w_pixel_vae:
                        closest_size, closest_ratio = get_closest_ratio(height, width, self.aspect_ratio)
                        closest_size = list(map(lambda x: int(x), closest_size))
                    
                        if closest_size[0] / height > closest_size[1] / width:
                            resize_size = closest_size[0], int(width * closest_size[0] / height)
                        else:
                            resize_size = int(height * closest_size[1] / width), closest_size[1]
                        
                        self.pixel_pil_transform = T.Compose([
                                    T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                                    T.CenterCrop(closest_size)])

                        self.pixel_transform = T.Compose([
                                                    T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                                                    T.CenterCrop(closest_size),
                                                    T.ToTensor(),
                                                    T.Normalize([.5], [.5]),
                                                        ])
                        
                        source_pixel_values = self.pixel_transform(source_im_pil)
                        source_pixel_images = self.pixel_pil_transform(source_im_pil)
                        source_pixel_grid_thw = torch.tensor([1, closest_size[0] // 8, closest_size[1] // 8])

                        edited_pixel_values = self.pixel_transform(target_im_pil)

                        tgt_h, tgt_w =  closest_size[0], closest_size[1]

                    source_image = source_im_pil.resize((seg_resized_width, seg_resized_height))
                    source_images = [get_image_info(source_image, self.image_min_pixel, self.image_max_pixel, seg_resized_width, seg_resized_height)]
                    
                    # construct sources
                    sources = {}
                    conversations = []

                    if sel_prompt in ['instruction_cn', 'instruction_enrich']:
                        res = '<image>' + random.choice(chinese_img2img_template).format(target_w=tgt_w, target_h=tgt_h, prompt=clean_prompt)
                    else:
                        res = '<image>' + random.choice(english_img2img_template).format(target_w=tgt_w, target_h=tgt_h, prompt=clean_prompt)
                    
                    
                    ## add hw indicate following the results 
                    if self.hw_indicator:
                        res = res + " " + f"<indicator:{tgt_h}>" + f"<indicator:{tgt_w}>"

                    conversations.append({'from': 'human', 'value': res}) 

                    rsp_img_tokens =  GENERATE_START_TOKEN 

                    if self.i2i_generate_method == 'line_break':
                        for i in range(tgt_h // self.vqvae_downscale):
                            rsp_img_tokens += DEFAULT_GENERATE_IMAGE_TOKEN * (tgt_w // self.vqvae_downscale)
                            rsp_img_tokens = rsp_img_tokens + EOL_TOKEN if  i != tgt_h // self.vqvae_downscale - 1 else rsp_img_tokens + GENERATE_END_TOKEN
                    elif self.i2i_generate_method == 'no_line':
                        for i in range(tgt_h // self.vqvae_downscale):
                            rsp_img_tokens += DEFAULT_GENERATE_IMAGE_TOKEN * (tgt_w // self.vqvae_downscale)
                        rsp_img_tokens += GENERATE_END_TOKEN
                    else:
                        raise f"only support t2i generate method in [line_break], but input method is {self.t2i_generate_method}"
                    
                    conversations.append({'from': 'gpt', 'value': rsp_img_tokens})

                    sources['conversations'] = conversations
                    sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=False))

                    all_input_ids = [] 
                    all_labels = []
                    all_semantic_values = []
                    all_image_grid_thw = []

                    # Qwen2.5-VL uses a default system message so I've added this.
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

                        if DEFAULT_IMAGE_TOKEN in user_input:
                            inputs = self.processor(text=[user_input], images=source_images, videos=None, padding=False, return_tensors='pt')
                            prompt_input_ids = inputs['input_ids']
                            
                            if self.w_pixel_vae:
                                start_index = (prompt_input_ids[0] == self.semantic_start_id).nonzero(as_tuple=True)[0].item()
                                end_index = (prompt_input_ids[0] == self.semantic_end_id).nonzero(as_tuple=True)[0].item()
                                
                                pixel_width, pixel_height = source_pixel_images.size
                                assert pixel_width * pixel_height % (16 * 16) == 0
                                
                                ## add line break 
                                num_pixel_tokens = PIXEL_START_TOKEN
                                for i in range(tgt_h // self.vqvae_downscale):
                                    num_pixel_tokens += DEFAULT_IMAGE_TOKEN * (tgt_w // self.vqvae_downscale)
                                    num_pixel_tokens = num_pixel_tokens + EOL_TOKEN if  i != tgt_h // self.vqvae_downscale - 1 else num_pixel_tokens + PIXEL_END_TOKEN
                                pixel_input_text =  num_pixel_tokens 

                                pixel_input_ids = self.processor.tokenizer(pixel_input_text, padding=False, return_tensors='pt')['input_ids']

                                if self.pixel_pad_left:
                                    prompt_input_ids = torch.cat((prompt_input_ids[:, :start_index], pixel_input_ids, prompt_input_ids[:, start_index:]), dim=1)
                                else:
                                    prompt_input_ids = torch.cat((prompt_input_ids[:, :end_index+1], pixel_input_ids, prompt_input_ids[:, end_index+1:]), dim=1)
                                
                            all_semantic_values.append(inputs['pixel_values'])
                            all_image_grid_thw.append(inputs[self.grid_key])
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


                    # print("edit sample length", input_ids.shape[0])
                    if input_ids.shape[0] > self.max_seq_length:
                        print("editing pass max length", input_ids.shape[0])
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
                        images=edited_pixel_values
                    )
                    data_dict[self.pil_image_key] = source_image

                    if self.semantic_key and self.grid_key:
                        semantic_values = torch.cat(all_semantic_values, dim=0)
                        image_thw = torch.cat(all_image_grid_thw, dim=0)
                        data_dict[self.semantic_key] = semantic_values
                        data_dict[self.grid_key] = image_thw

                    if self.w_pixel_vae:
                        data_dict.update({self.pixel_key: source_pixel_values, self.pixel_grid_key: source_pixel_grid_thw, self.generate_pixel_grid_key: source_pixel_grid_thw})

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
            if k not in (self.pixel_key, self.semantic_key, self.pil_image_key, 'images', self.grid_key):
                batched[k] = torch.stack(v, dim=0)
                
        return batched


if __name__ == "__main__":
    pass

