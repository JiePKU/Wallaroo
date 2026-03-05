# coding=utf-8
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = "6887"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import copy 
import wandb
import json
import glob
import random 
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import os
import json
import random
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision.transforms.functional import InterpolationMode
from diffusers.models import AutoencoderKL
from wallaroo.models import Wallaroo, pack_latents
from wallaroo.models.tokenizer_image import build_image_tokenizer
from wallaroo.datasets.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu, create_attention_mask_for_mmu_vit, create_attention_mask_for_mmu_vit_v1
from wallaroo.datasets import ImageNetDataset, UndDiscreteCaptionDataset, UndContinuousCaptionDataset, get_instruct_data_loader, Text2ImageDataset, CaptionMmuDataset
from wallaroo.datasets.und_caption_dataset import preprocess_plain, preprocess_multimodal
from wallaroo.optimizers import *
from wallaroo.schedulers.rf import timestep_transform, add_noise
from wallaroo.utils.simple_utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from wallaroo.datasets.constants import *
from wallaroo.datasets.utils import llava_to_openai, field_probabilities, center_crop_arr, get_image_info, smart_resize, get_closest_ratio
from qwen_vl_utils import process_vision_info

from transformers import AutoTokenizer, AutoProcessor
from transformers import CLIPImageProcessor

config = get_config()
vqvae_downscale = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(config.model.wallaroo.llm_model_path, cache_dir=config.model.wallaroo.cache_dir)
tokenizer = processor.tokenizer
tokenizer.bos_token_id = tokenizer.eos_token_id

special_tokens = (GENERATE_START_TOKEN, GENERATE_END_TOKEN, PIXEL_START_TOKEN, PIXEL_END_TOKEN, EOL_TOKEN, DEFAULT_GENERATE_IMAGE_TOKEN)
tokenizer.add_tokens(list(special_tokens)+hw_indicator_512_lst)

generate_start_id = processor.tokenizer.convert_tokens_to_ids(GENERATE_START_TOKEN)
generate_end_id = processor.tokenizer.convert_tokens_to_ids(GENERATE_END_TOKEN)
pixel_start_id = processor.tokenizer.convert_tokens_to_ids(PIXEL_START_TOKEN)
pixel_end_id = processor.tokenizer.convert_tokens_to_ids(PIXEL_END_TOKEN)
eol_id = processor.tokenizer.convert_tokens_to_ids(EOL_TOKEN)
image_pad_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
image_generate_pad_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_GENERATE_IMAGE_TOKEN)

semantic_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
semantic_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

resolution = config.resolution

def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

def is_english(char):
    return ('A' <= char <= 'Z') or ('a' <= char <= 'z')

def check_string(s):
    chinese_count = sum(is_chinese(c) for c in s)
    english_count = sum(is_english(c) for c in s)
    if chinese_count > 0 and english_count == 0:
        return '中文'
    elif english_count > 0 and chinese_count == 0:
        return '英文'
    elif chinese_count > 0 and english_count > 0:
        return '中英混合'
    else:
        return '其他'


def run_inference_on_gpu(rank, model, processor, vqvae_model, sample_prompts, output_path, output_path_grid, cfg_scale, image_token_count, img_nums_per_prompt):
    # Initialize distributed computing
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=torch.cuda.device_count())
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Move models to the current GPU
    model = model.to(device)
    vqvae_model = vqvae_model.to(device)
    model.eval()

    # Sample data from prompts
    total_prompts = len(sample_prompts)
    start_index = rank * (total_prompts // torch.cuda.device_count())
    end_index = (rank + 1) * (total_prompts // torch.cuda.device_count()) if rank != torch.cuda.device_count() - 1 else total_prompts

    grid_size = (2,2)

    # Iterate over sample prompts
    for metadata in tqdm(sample_prompts[start_index:end_index], desc=f'pid-[{str(rank)}]'):
        
        pid, prompt = metadata
        grid_imgs = []
        sample_count = 0

        for idx in range(img_nums_per_prompt):

            sample_path = output_path + f'/{pid}_{sample_count:05}.png'
            seed = int(torch.seed())

            tmp_template = random.choice(english_generate_template)

            gpt_rsp_template = "<|im_start|>assistant\n<|generate_start|>"

            messages = [{"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": tmp_template.format(target_w=resolution, target_h=resolution, prompt=prompt)},
                    ],
                },
            ]

            # Prepare inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            if config.hw_indicator:
                text = text + " " + f"<indicator:{resolution}>" + f"<indicator:{resolution}>"

            
            inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

            null_messages = [{"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": tmp_template.format(target_w=config.resolution, target_h=config.resolution, prompt='')},
                    ],
                },
            ]

            null_text = processor.apply_chat_template(null_messages, tokenize=False, add_generation_prompt=False)
            
            if config.hw_indicator:
                null_text = null_text + " " +  f"<indicator:{resolution}>" + f"<indicator:{resolution}>" 
            
            
            null_inputs = processor(text=[null_text], padding=True, return_tensors="pt").to(device)

            gpt_out = processor.tokenizer(gpt_rsp_template)
            inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor(gpt_out['input_ids'], device=device).unsqueeze(0)), dim=1)
            inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.tensor(gpt_out['attention_mask'], device=device).unsqueeze(0)), dim=1)

            null_inputs['input_ids'] = torch.cat((null_inputs['input_ids'], torch.tensor(gpt_out['input_ids'], device=device).unsqueeze(0)), dim=1)
            null_inputs = processor.tokenizer.pad(
                {'input_ids': [null_inputs['input_ids'].squeeze(0)]},
                max_length=inputs['input_ids'].shape[1],
                padding='max_length',
                return_attention_mask=True,
                padding_side='left',
                return_tensors='pt'
            ).to(device)

            try:
                ### for square image during stage 1, 2, and 3.1###
                # generated_ids = model.t2i_generate(inputs, null_inputs, vqvae_model, eot_token=tokenizer.eos_token_id, max_new_tokens=2048, top_k=1000, top_p=1.0, cfg=cfg_scale, generate_start_id=generate_start_id, generate_end_id=generate_end_id, img_token_count=image_token_count).squeeze(0)
                
                ### for multi-resolution image during stage 3.2 and 4 ###
                generated_ids = model.mar_t2i_generate(inputs, null_inputs, vqvae_model, eot_token=tokenizer.eos_token_id, top_k=1000, top_p=0.95, cfg=cfg_scale, generate_start_id=generate_start_id, generate_end_id=generate_end_id, eol_id=eol_id, tgt_h = resolution, tgt_w=resolution, ds = vqvae_downscale).squeeze(0)
                output = vqvae_model.decode_code(
                    generated_ids, torch.Size([1, 8, resolution // 16, resolution // 16])
                )

                # Postprocessing
                output = F.interpolate(output, size=[resolution, resolution], mode='bicubic').permute(0, 2, 3, 1)[0]
                sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            
                sample = Image.fromarray(sample)
                sample.save(sample_path)
                sample_count += 1
                grid_imgs.append(sample)

            except Exception as e:
                print(f'Error: {e}!!!')
                continue
        

        sample_grid_path = output_path_grid + f'/{pid}.png'

        img_size = grid_imgs[0].size

        total_width = grid_size[1] * img_size[0]
        total_height = grid_size[0] * img_size[1]

        new_im = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

        for index, img in enumerate(grid_imgs):
            row = index // grid_size[1]
            col = index % grid_size[1]
            x = col * img_size[0]
            y = row * img_size[1]
            new_im.paste(img, (x, y))
        
        new_im.save(sample_grid_path)
        grid_imgs = []



if __name__ == '__main__':
    weight_dtype = torch.bfloat16
    model = Wallaroo(**config.model.wallaroo).to(weight_dtype)
    model.eval()

    model.wallaroo.config.pixel_start_token_id = pixel_start_id
    model.wallaroo.config.pixel_end_token_id = pixel_end_id
    model.wallaroo.config.generate_start_token_id = generate_start_id
    model.wallaroo.config.generate_end_token_id = generate_end_id
    model.wallaroo.config.generate_img_pad_id = image_generate_pad_id
    
    resume_checkpoint = torch.load(config.pretrained_path, map_location="cpu")
    m, u = model.load_state_dict(resume_checkpoint['state_dict'], strict=False)
    print(f"Transformer model missing keys: {len(m)}, unexpected keys: {len(u)}")
    assert len(u) == 0, f"Found unexpected keys: {u}, please check the checkpoint carefully."
    del resume_checkpoint

    # VQ-VAE model for t2i image-encoding
    vqvae_config = config.model.vqvae_model
    vqvae_model = build_image_tokenizer(vqvae_config)
    vqvae_model.eval()
    vqvae_model.requires_grad_(False)

    dpg_csv_name = '/dpg_bench/dpg_bench.csv'
    import pandas as pd 
    df = pd.read_csv(dpg_csv_name)
    all_prompts = []
    unique_prompt = {}
    for i in range(df.shape[0]):
        item_id = df.iloc[i]['item_id']
        prompt_text = df.iloc[i]['text']
        prompt_text = prompt_text.strip().strip('"').strip("'")
        if item_id in unique_prompt: continue
        unique_prompt[ item_id ] = 1
        all_prompts.append( (item_id, prompt_text )  )

    sample_prompts = all_prompts

    output_path = f"{config.save_path}_{config.cfg}"
    os.makedirs(output_path, exist_ok=True)

    output_path_grid = f"{config.save_path}_grid_{config.cfg}"
    os.makedirs(output_path_grid, exist_ok=True )

    cfg_scale = config.cfg
    image_token_count = config.image_token_count
    img_nums_per_prompt = 4
    world_size = torch.cuda.device_count()

    # Multi-GPU Inference with torch.multiprocessing.spawn
    torch.multiprocessing.spawn(run_inference_on_gpu, args=(model, processor, vqvae_model, sample_prompts, output_path, output_path_grid, cfg_scale, image_token_count, img_nums_per_prompt), nprocs=world_size)

        
        
        
