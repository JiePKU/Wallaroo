# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import random
from typing import Any, Callable, Optional
import copy
import traceback
import collections
import numpy as np
import torchvision.transforms as T

import torch
from torchvision.datasets.folder import DatasetFolder, default_loader
from wallaroo.utils.simple_utils import image_transform
from .constants import *
from .utils import llava_to_openai, field_probabilities, center_crop_arr, get_image_info, smart_resize, get_closest_ratio


class ImageNetDataset(DatasetFolder):
    def __init__(
        self,
        root: str,
        processor, max_seq_length,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        image_size=256,
        vqvae_downscale=16, hw_indicator=False, t2i_generate_method='line_break', **kwargs
    ):
        IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

        self.transform = image_transform
        self.image_size = self.resolution = image_size

        self.uncond_p = kwargs.get('uncond_p', 0.1)
        self.cn_prompt_sample_prob = kwargs.get('cn_prompt_sample_prob', 0.5)
        self.processor = processor
        self.vqvae_downscale = vqvae_downscale
        self.max_seq_length = max_seq_length
        self.t2i_generate_method = t2i_generate_method
        self.hw_indicator = hw_indicator

        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=self.transform,
            target_transform=None,
            is_valid_file=is_valid_file,
        )

        with open('./wallaroo/datasets/imagenet_label_mapping', 'r') as f:
            self.cn_labels, self.en_labels = {}, {}
            for l in f:
                num, english_description, chinese_description = l.split(":")
                self.en_labels[int(num)] = english_description.strip()
                self.cn_labels[int(num)] = chinese_description.strip()

        print("ImageNet dataset loaded.")


    def __getitem__(self, idx):
        try:
            path, target = self.samples[idx]
            image = self.loader(path)
            image = self.transform(image, resolution=self.image_size)

            if random.random() < 0.5:
                clean_prompt = self.en_labels[target]
                sel_prompt = 'prompt_en'
            else:
                clean_prompt = self.cn_labels[target]
                sel_prompt = 'prompt_cn'
            
            if random.random() < self.uncond_p:
                clean_prompt = ''
                
            tgt_h, tgt_w =  self.resolution, self.resolution

            target_grid_thw = torch.tensor([1, tgt_h // (self.vqvae_downscale // 2), tgt_w // (self.vqvae_downscale // 2)])

            # construct sources
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
            # Qwen2.5-VL does not use them
            input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
            labels = torch.cat(all_labels, dim=0).to(torch.long)


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
                images=image,
                image_grid_thw=target_grid_thw
            )

            return data_dict

        except Exception as e:
            print(e)
            return self.__getitem__(idx+1)

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('images'):
                batched[k] = torch.stack(v, dim=0)

        return batched


if __name__ == '__main__':
    pass
