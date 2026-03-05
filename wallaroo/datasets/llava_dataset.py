import os 
import pickle 
import json 
import torch 
import glob 
import numpy as np 
import random
import torch.distributed as dist
import copy
import traceback
import collections
import torchvision.transforms as T

from io import BytesIO
from PIL import Image 
from functools import partial
from typing import List, Dict
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
from transformers import AutoTokenizer, CLIPImageProcessor

from data_curation import TFReader
from wallaroo.datasets.llava import conversation as conversation_lib

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."

def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_v0(
        sources,
        tokenizer,
):
    # Let's assume has_image is false, since we will process the image token separately
    has_image = False

    # Adapted from llava-phi/mipha/train/train.py
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX


        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            # if has_image:
            #     round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # +1 for <|endoftext|>
            #     instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1   # -1 for <image>
            # else:
            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )

class LLavaDataset(torch.utils.data.Dataset):
    def __init__(self, data_urls, tokenizer):
        super(LLavaDataset, self).__init__()
        self.tokenizer = tokenizer
        
        # generate dataset
        data_paths = data_urls.split(":")
        print("all data paths: ", data_paths)
        data_files = []
        for data_path in data_paths:
            data_file = glob.glob(os.path.join(data_path, "*.tfrecord"))
            data_files.extend(data_file)

        random.shuffle(data_files)
        # read tfrecord
        self.desc_dict = {'state_shape': 'byte', 'state_dict': 'byte', 'meta': 'byte'}  
        self.list_data_dict = DatasetMem(cfg=None, data_file=data_files, desc_dict=self.desc_dict)
        self.length = len(self.list_data_dict)
        self.ori_imgs_nums = self.length
        print("ori_imgs_nums: ", self.ori_imgs_nums)

        self.prompt_list = [
                                "Describe this picture in detail.",
                                "What do you see in this image?",
                                "Can you provide a detailed description of this photo?",
                                "Give a comprehensive description of the contents of this image.",
                                "Explain the scene depicted in this picture.",
                                "Identify and describe the main elements in this image.",
                                "What are the key features of this photo?",
                                "Please describe what is happening in this image.",
                                "Summarize the visual elements present in this picture.",
                                "What can you tell about the context of this image?",
                                "Provide a detailed explanation of the objects and actions in this image.",
                                "How would you describe the setting and characters in this picture?",
                                "What are the notable details in this photo?",
                                "Please give a full description of the scene shown in this image.",
                                "What is the composition of this image?",
                                "Can you describe the mood and atmosphere of this picture?",
                                "List and describe the elements you observe in this image.",
                                "Describe the interactions between objects in this photo.",
                                "What story does this image tell?",
                                "Give an accurate and detailed description of this picture."
                            ]

        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-imagen-hl/hadoop-imagen/huggingface_models/')

        idx = 0
        data = self.list_data_dict[idx]
        meta = json.loads(bytes(data['meta']).decode())
        raw_dict = pickle.loads(data['state_dict'])

        buffer = BytesIO(raw_dict['raw_image'])
        im_pil = Image.open(buffer)
        if im_pil.mode != 'RGB':
            im_pil = im_pil.convert('RGB')

        image = self.processor.preprocess(im_pil, return_tensors='pt')['pixel_values'][0]
        # image = self.transform(im_pil)

        caption = meta.get('variables').get('internLM_caption__english_detailed') 
        prompt = random.choice(self.prompt_list)

        # 构造sources
        sources, conversations = {}, {}
        conversations = []
        conversations.append({'from': 'human', 'value': f"{prompt}\n<image>"})
        conversations.append({'from': 'gpt', 'value': caption})
        sources['conversations'] = conversations
        sources = [sources]
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        
        data_dict = preprocess_v0(sources, self.tokenizer)

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0],
                             image=image)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        data = self.list_data_dict[idx]
        meta = json.loads(bytes(data['meta']).decode())
        raw_dict = pickle.loads(data['state_dict'])

        try:
            
                
            buffer = BytesIO(raw_dict['raw_image'])
            im_pil = Image.open(buffer)
            if im_pil.mode != 'RGB':
                im_pil = im_pil.convert('RGB')
            
            image = self.processor.preprocess(im_pil, return_tensors='pt')['pixel_values'][0]
            
            caption = meta.get('variables').get('internLM_caption__english_detailed') 
            prompt = random.choice(self.prompt_list)
            # 构造sources
            sources, conversations = {}, {}
            conversations = []
            conversations.append({'from': 'human', 'value': f"{prompt}\n<image>"})
            conversations.append({'from': 'gpt', 'value': caption})
            sources['conversations'] = conversations
            sources = [sources]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
            
            data_dict = preprocess_v0(sources, self.tokenizer)

            data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0],
                                input_ids_system=data_dict["input_ids_system"][0],
                                image=image)

            return data_dict
        except Exception as e:
            print(f"error in data loading: {e}")
            return self.ratio_index[closest_ratio]

def collate_fn(
        instances,
        tokenizer=None,
        max_length=77,
):
    input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "input_ids_system"))
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    input_ids_system = torch.stack(input_ids_system, dim=0)

    offset = max_length - input_ids.shape[-1] - input_ids_system.shape[-1]

    if input_ids.shape[-1] < max_length - input_ids_system.shape[-1]:
        pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, pad_tube], dim=1)

        pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
        labels = torch.cat([labels, pad_tube], dim=1)

    min_max_len = min(
        max_length - input_ids_system.shape[-1],
        tokenizer.model_max_length - input_ids_system.shape[-1],
    )

    input_ids = input_ids[:, :min_max_len]
    labels = labels[:, :min_max_len]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        input_ids_system=input_ids_system,
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

    return batch

def get_instruct_data_loader(
        data_urls,
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
):
    train_dataset = LLavaDataset(data_urls, tokenizer)
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader

if __name__ =="__main__":
    pass