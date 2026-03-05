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
from typing import List, Dict

from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
from transformers import AutoTokenizer, CLIPImageProcessor

from data_curation import TFReader
from wallaroo.datasets.llava import conversation as conversation_lib
from wallaroo.utils.dist_utils import get_world_size, clip_grad_norm_, get_local_rank, get_rank

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]

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


def preprocess_plain(sources, tokenizer):
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        # source[0]['value'] = DEFAULT_IMAGE_TOKEN
        source[0]['value'] = ""
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)

    # tokenize conversations
    # input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    input_ids = [tokenizer(prompt)["input_ids"] + [tokenizer.eos_token_id] for prompt in conversations]
    targets = copy.deepcopy(input_ids)

    for target, source in zip(targets, sources):
        # tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        tokenized_len = len(tokenizer(source[0]['value'])["input_ids"])
        if tokenized_len > 0:
            target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(targets))

class UndDiscreteCaptionDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_urls, max_seq_length, resolution, tokenizer):
        super(UndDiscreteCaptionDataset, self).__init__()
        self.max_seq_length = max_seq_length
        self.resolution = resolution
        self.tokenizer = tokenizer
        
        # generate dataset
        data_paths = data_urls.split(":")
        print("all data paths: ", data_paths)
        self.data_files = []
        for data_path in data_paths:
            data_file = glob.glob(os.path.join(data_path, "*.tfrecord"))
            self.data_files.extend(data_file)

        random.shuffle(self.data_files)
        
        self.ori_imgs_nums = -1
        print("ori_imgs_nums: ", self.ori_imgs_nums)

        self.transform = T.Compose([
                T.Resize(self.resolution, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        world_size = get_world_size()
        rank = get_rank()
        self.start = 0
        self.end = len(self.data_files)
        if worker_info is None:  # 单进程加载数据
            iter_start = self.start
            iter_end = self.end
        else:   # 多进程加载数据，分割数据集
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

                    buffer = BytesIO(raw_dict['raw_image'])

                    im_pil = Image.open(buffer)
                    if im_pil.mode != 'RGB':
                        im_pil = im_pil.convert('RGB')
                    
                    image = self.transform(im_pil)
                    
                    prompt_fields = []
                    if meta.get('variables').get('image_caption', ''):
                        prompt_fields.append(meta.get('variables').get('image_caption'))
                    if meta.get('variables').get('vllm_caption', []):
                        vllm_caption = eval(meta.get('variables').get('vllm_caption', [])[0])
                        if vllm_caption.get('English_brief', ''):
                            prompt_fields.append(vllm_caption.get('English_brief', ''))
                        if vllm_caption.get('English_detailed', ''):
                            prompt_fields.append(vllm_caption.get('English_detailed', ''))
                        # if vllm_caption.get('Chinese_brief', ''):
                        #     prompt_fields.append(vllm_caption.get('Chinese_brief', ''))
                        # if vllm_caption.get('Chinese_detailed', ''):
                        #     prompt_fields.append(vllm_caption.get('Chinese_detailed', ''))
                    if not prompt_fields:
                        continue
                    
                    prompt_caption = random.choice(prompt_fields)

                    yield {'images': image, 'input_ids': prompt_caption}
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
            if k not in ('input_ids'):
                batched[k] = torch.stack(v, dim=0)

        return batched

class UndContinuousCaptionDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_urls, max_seq_length, resolution, tokenizer):
        super(UndContinuousCaptionDataset, self).__init__()
        self.max_seq_length = max_seq_length
        self.resolution = resolution
        self.tokenizer = tokenizer
        
        # generate dataset
        data_paths = data_urls.split(":")
        print("all data paths: ", data_paths)
        self.data_files = []
        for data_path in data_paths:
            data_file = glob.glob(os.path.join(data_path, "*.tfrecord"))
            self.data_files.extend(data_file)

        random.shuffle(self.data_files)
        self.ori_imgs_nums = -1
        print("ori_imgs_nums: ", self.ori_imgs_nums)

        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-imagen-hl/hadoop-imagen/huggingface_models/')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        world_size = get_world_size()
        rank = get_rank()
        self.start = 0
        self.end = len(self.data_files)
        if worker_info is None:  # 单进程加载数据
            iter_start = self.start
            iter_end = self.end
        else:   # 多进程加载数据，分割数据集
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

                    buffer = BytesIO(raw_dict['raw_image'])

                    im_pil = Image.open(buffer)
                    if im_pil.mode != 'RGB':
                        im_pil = im_pil.convert('RGB')
                    
                    image = self.processor.preprocess(im_pil, return_tensors='pt')['pixel_values'][0]
                    
                    prompt_fields = []
                    if meta.get('variables').get('image_caption', ''):
                        prompt_fields.append(meta.get('variables').get('image_caption'))
                    if meta.get('variables').get('vllm_caption', []):
                        vllm_caption = eval(meta.get('variables').get('vllm_caption', [])[0])
                        if vllm_caption.get('English_brief', ''):
                            prompt_fields.append(vllm_caption.get('English_brief', ''))
                        if vllm_caption.get('English_detailed', ''):
                            prompt_fields.append(vllm_caption.get('English_detailed', ''))
                    if not prompt_fields:
                        continue
                    
                    caption = random.choice(prompt_fields)

                    # 构造sources
                    sources, conversations = {}, {}
                    conversations = []
                    conversations.append({'from': 'human', 'value': f"null\n<image>"}) # fake format
                    conversations.append({'from': 'gpt', 'value': caption})
                    sources['conversations'] = conversations
                    sources = [sources]
                    sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

                    data_dict = preprocess_plain(sources, self.tokenizer)

                    data_dict = dict(input_ids=data_dict["input_ids"][0],
                                        labels=data_dict["labels"][0],
                                        image=image)
                    yield data_dict
                except StopIteration:
                    break
                except Exception as e:
                    print(f"error in data loading: {e}")
                    traceback.print_exc()
                    continue
        
            
    def collate_fn(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                              for key in ("input_ids", "labels"))
                              
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)

        if input_ids.shape[-1] < self.max_seq_length:
            offset = self.max_seq_length - input_ids.shape[-1]
            pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, pad_tube], dim=1)

            offset = self.max_seq_length - labels.shape[-1]
            pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
            labels = torch.cat([labels, pad_tube], dim=1)

        min_max_len = min(self.max_seq_length, self.tokenizer.model_max_length)

        input_ids = input_ids[:, :min_max_len]
        labels = labels[:, :min_max_len]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch
            
if __name__ =="__main__":
    pass