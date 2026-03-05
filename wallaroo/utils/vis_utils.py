import os
import json
import copy
import logging
import math
import shutil
import time
import datetime
import random 
import glob
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import wandb
import torch
import glob
import gc
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
from typing import Union
from mmcv.runner import LogBuffer
from torch.distributions import LogisticNormal
from diffusers.models import AutoencoderKL
from torch.optim import AdamW
from torchvision.transforms.functional import InterpolationMode
from pytorch_lightning.utilities import CombinedLoader

from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer, AutoProcessor

from wallaroo.models import pack_latents
from wallaroo.datasets.utils import llava_to_openai, field_probabilities, center_crop_arr, get_image_info, smart_resize, get_closest_ratio
from wallaroo.datasets.constants import *
from wallaroo.optimizers import *
from wallaroo.utils.logging import check_string
from wallaroo.datasets.constants import *

@torch.no_grad()
def prepare_mmu_infer_inputs(processor, config, img_path, device):

    preproc_config = config.dataset.preprocessing

    size_factor = preproc_config.SEG_IMAGE_FACTOR
    image_min_pixel = preproc_config.SEG_MIN_PIXELS
    image_max_pixel = preproc_config.SEG_MAX_PIXELS
    max_ratio = preproc_config.MAX_RATIO
    grid_key = "image_grid_thw"

    image = Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 构造sources
    width, height = image.size
    seg_resized_height, seg_resized_width = smart_resize(height, width, factor=size_factor, min_pixels=image_min_pixel, max_pixels=image_max_pixel, max_ratio=max_ratio)

    image = image.resize((seg_resized_width, seg_resized_height))
    images = [get_image_info(image, image_min_pixel, image_max_pixel, seg_resized_width, seg_resized_height)]
    
    
    # 构造sources
    sources = {}
    conversations = []

    conversations.append({'from': 'human', 'value': '<image>' + "请详细描述这张图片"}) 
    conversations.append({'from': 'gpt', 'value': ''})
    
    sources['conversations'] = conversations
    sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=False))

    all_input_ids = [] 
    all_labels = []
    all_semantic_values = []
    all_image_grid_thw = []

    # Qwen2-VL uses a default system message so I've added this.
    if len(SYSTEM_MESSAGE) > 0:
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
        system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
        
        all_input_ids.append(system_message_input_ids.squeeze(0))
        all_labels.append(system_labels.squeeze(0))

    for _, j in enumerate(range(0, len(sources), 2)):
        user_input = sources[j]
        gpt_response = sources[j + 1]
        
        user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
        gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

        if DEFAULT_IMAGE_TOKEN in user_input:
            inputs = processor(text=[user_input], images=images, videos=None, padding=False, return_tensors='pt')
            prompt_input_ids = inputs['input_ids']

            all_semantic_values.append(inputs['pixel_values'])
            all_image_grid_thw.append(inputs[grid_key])

        else:
            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        input_ids = prompt_input_ids.squeeze(0)

        all_input_ids.append(input_ids)

    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long).to(device)
    inputs['input_ids'] = input_ids.unsqueeze(0).to(device)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids']).to(device)
    inputs['pixel_values'] = [torch.cat(all_semantic_values, dim=0).to(device)]
    inputs['image_grid_thw'] = [torch.cat(all_image_grid_thw, dim=0).to(device)]

    return inputs

@torch.no_grad()
def visualize_mmu_predictions(model, processor, config, global_step, device):
    model.eval()

    imgs_path = glob.glob(config.dataset.params.mmu_validation_imgs)
    test_imgs_path = random.sample(imgs_path, k=4)

    pil_images, pred_texts = [], []
    for test_img_path in test_imgs_path:
        mmu_infer_inputs = prepare_mmu_infer_inputs(processor, config, test_img_path, device)
        generated_ids = model.wallaroo.generate(**mmu_infer_inputs, max_new_tokens=256)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(mmu_infer_inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        pil_images.append(Image.open(test_img_path))
        pred_texts.append(output_text)
    
    wandb_images = [wandb.Image(image, caption=f'pred_caption: {pred_text}') for i, (image, pred_text) in
                    enumerate(zip(pil_images, pred_texts))]
    wandb.log({"mmu visualize": wandb_images}, step=global_step)
    model.train()

@torch.no_grad()
def visualize_t2i_predictions_1d(model, vqvae_model, processor, config, global_step, device, image_token_count=576, max_new_tokens=1024, target_resolution=384):
    model.eval()

    generate_start_id = processor.tokenizer.convert_tokens_to_ids(GENERATE_START_TOKEN)
    generate_end_id = processor.tokenizer.convert_tokens_to_ids(GENERATE_END_TOKEN)

    tgt_h = tgt_w = target_resolution

    f = open(config.dataset.params.validation_prompts_file, 'r')
    lines = f.readlines()

    text_prompts, pil_images = [], []

    prompts = random.sample(lines, k=4)

    for prompt in prompts:
        try:
            if prompt.startswith('"') and prompt.endswith('"'):
                prompt = prompt.strip('"')

            if check_string(prompt) == '英文':
                tmp_template = random.choice(english_generate_template)
            else:
                tmp_template = random.choice(chinese_generate_template)

            gpt_rsp_template = "<|im_start|>assistant\n<|generate_start|>"

            messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": tmp_template.format(target_w=tgt_w, target_h=tgt_h, prompt=prompt)},
                            ],
                        },
                    ]
                    
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(
                    text=[text_prompt],
                    padding=True,
                    return_tensors="pt",
                )
            inputs = inputs.to(device)

            null_messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": tmp_template.format(target_w=tgt_w, target_h=tgt_h, prompt='')},
                            ],
                        },
                    ]
            null_text = processor.apply_chat_template(null_messages, tokenize=False, add_generation_prompt=False)
            null_inputs = processor(
                text=[null_text],
                padding=True,
                return_tensors="pt",
            ).to(device)

            gpt_out = processor.tokenizer(gpt_rsp_template)
            inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor(gpt_out['input_ids'], device=device).unsqueeze(0)), dim=1)
            inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.tensor(gpt_out['attention_mask'], device=device).unsqueeze(0)), dim=1)

            null_inputs['input_ids'] = torch.cat((null_inputs['input_ids'], torch.tensor(gpt_out['input_ids'], device=device).unsqueeze(0)), dim=1)
            null_inputs = processor.tokenizer.pad(
                                    {'input_ids': [null_inputs['input_ids'].squeeze(0)]},
                                    max_length=inputs['input_ids'].shape[1],
                                    padding='max_length',
                                    # truncation=True,
                                    return_attention_mask=True,
                                    padding_side='left',
                                    return_tensors='pt').to(device)
            generated_ids = model.t2i_generate(inputs, null_inputs, vqvae_model, eot_token=processor.tokenizer.eos_token_id, max_new_tokens=2048, top_k=1000, top_p=1.0, cfg=config.training.guidance_scale, generate_start_id=generate_start_id, generate_end_id=generate_end_id, img_token_count=image_token_count).squeeze(0)
            output = vqvae_model.decode_code(generated_ids, torch.Size([1, config.model.wallaroo.codebook_embed_dim, tgt_h // 16, tgt_w // 16])) # output value is between [-1, 1]
            # postprocess
            output = F.interpolate(output, size=[tgt_h, tgt_w], mode='bicubic').permute(0, 2, 3, 1)[0]
            sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            pil_image = Image.fromarray(sample)
        except Exception as e:
            print(e)
            pil_image = Image.new("RGB", (tgt_w, tgt_h), (0, 0, 0))

        pil_images.append(pil_image)
        text_prompts.append(prompt)

    wandb_images = [wandb.Image(image, caption=f'text_prompt: {text_prompt}') for i, (image, text_prompt) in
                    enumerate(zip(pil_images, text_prompts))]

    wandb.log({"t2i visualize": wandb_images}, step=global_step)

    model.train()



@torch.no_grad()
def visualize_mar_t2i_predictions_1d(model, vqvae_model, processor, config, global_step, device, vqvae_downscale, hw_indicator=False):
    model.eval()

    generate_start_id = processor.tokenizer.convert_tokens_to_ids(GENERATE_START_TOKEN)
    generate_end_id = processor.tokenizer.convert_tokens_to_ids(GENERATE_END_TOKEN)
    eol_id = processor.tokenizer.convert_tokens_to_ids(EOL_TOKEN)
    eos_token_id = eot_token=processor.tokenizer.eos_token_id

    f = open(config.dataset.params.validation_prompts_file, 'r')
    lines = f.readlines()

    text_prompts, pil_images = [], []

    prompts = random.sample(lines, k=4)

    for prompt in prompts:
        try:
            tgt_h, tgt_w = ASPECT_RATIO_512[random.choice(list(ASPECT_RATIO_512.keys()))]

            if prompt.startswith('"') and prompt.endswith('"'):
                prompt = prompt.strip('"')

            if check_string(prompt) == '英文':
                tmp_template = random.choice(english_generate_template)
            else:
                tmp_template = random.choice(chinese_generate_template)
            gpt_rsp_template = "<|im_start|>assistant\n<|generate_start|>"

            messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": tmp_template.format(target_w=tgt_w, target_h=tgt_h, prompt=prompt)},
                            ],
                        },
                    ]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            if hw_indicator:
                text_prompt = text_prompt + " " + f"<indicator:{tgt_h}>" + f"<indicator:{tgt_w}>"

            inputs = processor(
                    text=[text_prompt],
                    padding=True,
                    return_tensors="pt",
                )
            inputs = inputs.to(device)

            null_messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": tmp_template.format(target_w=tgt_w, target_h=tgt_h, prompt='')},
                            ],
                        },
                    ]
            null_text = processor.apply_chat_template(null_messages, tokenize=False, add_generation_prompt=False)

            if hw_indicator:
                # null_text = null_text + " " + f"<indicator:{tgt_h//vqvae_downscale}>" + f"<indicator:{tgt_w//vqvae_downscale}>"
                null_text = null_text + " " + f"<indicator:{tgt_h}>" + f"<indicator:{tgt_w}>" 

            null_inputs = processor(
                text=[null_text],
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            gpt_out = processor.tokenizer(gpt_rsp_template)
            
            inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor(gpt_out['input_ids'], device=device).unsqueeze(0)), dim=1)
            inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.tensor(gpt_out['attention_mask'], device=device).unsqueeze(0)), dim=1)

            null_inputs['input_ids'] = torch.cat((null_inputs['input_ids'], torch.tensor(gpt_out['input_ids'], device=device).unsqueeze(0)), dim=1)
            null_inputs = processor.tokenizer.pad(
                                    {'input_ids': [null_inputs['input_ids'].squeeze(0)]},
                                    max_length=inputs['input_ids'].shape[1],
                                    padding='max_length',
                                    # truncation=True,
                                    return_attention_mask=True,
                                    padding_side='left',
                                    return_tensors='pt').to(device)
            
            generated_ids = model.mar_t2i_generate(inputs, null_inputs, vqvae_model, eot_token=eos_token_id, top_k=1000, top_p=1.0, cfg=config.training.guidance_scale, generate_start_id=generate_start_id, generate_end_id=generate_end_id, eol_id=eol_id, tgt_h = tgt_h, tgt_w=tgt_w, ds = vqvae_downscale).squeeze(0)
            
            output = vqvae_model.decode_code(generated_ids, torch.Size([1, config.model.wallaroo.codebook_embed_dim, tgt_h // 16, tgt_w // 16])) # output value is between [-1, 1]
            # postprocess
            output = F.interpolate(output, size=[tgt_h, tgt_w], mode='bicubic').permute(0, 2, 3, 1)[0]
            sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            pil_image = Image.fromarray(sample)
        except Exception as e:
            print(e)
            pil_image = Image.new("RGB", (tgt_w, tgt_h), (0, 0, 0))

        pil_images.append(pil_image)
        text_prompts.append(prompt)

    wandb_images = [wandb.Image(image, caption=f'text_prompt: {text_prompt}') for i, (image, text_prompt) in
                    enumerate(zip(pil_images, text_prompts))]

    wandb.log({"t2i visualize": wandb_images}, step=global_step)

    model.train()



def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # # generate the binary mask: 0 is keep, 1 is remove
    # mask = torch.zeros([N, L], device=x.device)
    # mask[:, :len_keep] = 1
    # # unshuffle to get the binary mask
    # mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, ids_restore