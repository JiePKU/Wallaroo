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

@torch.no_grad()
def prepare_edit_infer_inputs(model, pixel_vae_model, processor, config, image, instruction, tmp_template, device, hw_indicator=True):


    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    wallaroo_config = config.model.wallaroo

    size_factor = preproc_config.SEG_IMAGE_FACTOR
    image_min_pixel = preproc_config.SEG_MIN_PIXELS
    image_max_pixel = preproc_config.SEG_MAX_PIXELS
    max_ratio = preproc_config.MAX_RATIO
    w_pixel_vae = config.model.wallaroo.w_pixel_vae
    pixel_pad_left = config.model.wallaroo.pixel_pad_left

    grid_key = "image_grid_thw"
    semantic_key = "semantic_values"
    pixel_key = "pixel_values"
    pixel_grid_key = "pixel_grid_thw"
    pil_image_key = 'pil_image'
    weight_dtype = torch.bfloat16

    semantic_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    semantic_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    aspect_ratio = eval('ASPECT_RATIO_512')

    # construct sources
    width, height = image.size
    seg_resized_height, seg_resized_width = smart_resize(height, width, factor=size_factor, min_pixels=image_min_pixel, max_pixels=image_max_pixel, max_ratio=max_ratio)

    if w_pixel_vae:
        closest_size, closest_ratio = get_closest_ratio(height, width, aspect_ratio)
        closest_size = list(map(lambda x: int(x), closest_size))
    
        if closest_size[0] / height > closest_size[1] / width:
            resize_size = closest_size[0], int(width * closest_size[0] / height)
        else:
            resize_size = int(height * closest_size[1] / width), closest_size[1]
        
        pixel_pil_transform = T.Compose([
                    T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                    T.CenterCrop(closest_size)])

        pixel_transform = T.Compose([
                                    T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                                    T.CenterCrop(closest_size),
                                    T.ToTensor(),
                                    T.Normalize([.5], [.5]),
                                        ])
        
        pixel_values = pixel_transform(image)
        pixel_images = pixel_pil_transform(image)

        pixel_grid_thw = torch.tensor([1, closest_size[0] // 8, closest_size[1] // 8])


    image = image.resize((seg_resized_width, seg_resized_height))
    images = [get_image_info(image, image_min_pixel, image_max_pixel, seg_resized_width, seg_resized_height)]
    
    tgt_h, tgt_w =  closest_size[0], closest_size[1]

    # construc sources
    sources = {}
    conversations = []

    res = '<image>' + tmp_template.format(target_w=tgt_w, target_h=tgt_h, prompt=instruction)
    
    ## add hw indicate following the results 
    if hw_indicator:
        res = res + " " + f"<indicator:{tgt_h}>" + f"<indicator:{tgt_w}>"

    conversations.append({'from': 'human', 'value': res}) 
    conversations.append({'from': 'gpt', 'value': ''})
    
    sources['conversations'] = conversations
    sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=False))

    all_input_ids = [] 
    all_labels = []
    all_semantic_values = []
    all_image_grid_thw = []
    all_second_gird = []

    # Qwen2.5-VL uses a default system message so I've added this.
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

            if w_pixel_vae:
                start_index = (prompt_input_ids[0] == semantic_start_id).nonzero(as_tuple=True)[0].item()
                end_index = (prompt_input_ids[0] == semantic_end_id).nonzero(as_tuple=True)[0].item()
                
                pixel_width, pixel_height = pixel_images.size
                assert pixel_width * pixel_height % (16 * 16) == 0
                
                ## add line break 
                num_pixel_tokens = PIXEL_START_TOKEN
                for i in range(tgt_h // 16):
                    num_pixel_tokens += DEFAULT_IMAGE_TOKEN * (tgt_w // 16)
                    num_pixel_tokens = num_pixel_tokens + EOL_TOKEN if  i != tgt_h // 16 - 1 else num_pixel_tokens + PIXEL_END_TOKEN

                pixel_input_text =  num_pixel_tokens 
                
                pixel_input_ids = processor.tokenizer(pixel_input_text, padding=False, return_tensors='pt')['input_ids']
                if pixel_pad_left:
                    prompt_input_ids = torch.cat((prompt_input_ids[:, :start_index], pixel_input_ids, prompt_input_ids[:, start_index:]), dim=1)
                else:
                    prompt_input_ids = torch.cat((prompt_input_ids[:, :end_index+1], pixel_input_ids, prompt_input_ids[:, end_index+1:]), dim=1)
                
            all_semantic_values.append(inputs['pixel_values'])
            all_image_grid_thw.append(inputs[grid_key])
        else:
            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        input_ids = prompt_input_ids.squeeze(0)

        all_input_ids.append(input_ids)
    
    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long).to(device)
    inputs['input_ids'] = input_ids.unsqueeze(0).to(device)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids']).to(device)
    inputs['pixel_values'] = [torch.cat(all_semantic_values, dim=0).to(device)]
    inputs['image_grid_thw'] = torch.cat(all_image_grid_thw, dim=0).to(device)

    if w_pixel_vae:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                z_q, _, [_, z, raw_indice] = pixel_vae_model.encode(pixel_values.to(device, non_blocking=True).unsqueeze(0))
                z = z.reshape(z.shape[0], -1, z.shape[3]).to(device).to(weight_dtype) # unquanted vector  z shape b h w c

            vae_features = model.pixel_adapter(z)

        inputs['pixel_vae_values'] = [vae_features.squeeze(0)]
        inputs[pixel_grid_key] = pixel_grid_thw.unsqueeze(0).to(device)

    return inputs, tgt_h, tgt_w

from PIL import Image


@torch.no_grad()
def run_inference_on_gpu(rank, model, processor, pixel_vae_model, vqvae_model, sample_prompts, data, dir_path, output_path, edited_path, cfg_scale, image_token_count):

    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=torch.cuda.device_count())
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Move models to the current GPU
    model = model.to(device)
    vqvae_model = vqvae_model.to(device)
    pixel_vae_model = pixel_vae_model.to(device)
    model.eval()

    # Sample data from prompts
    total_prompts = len(sample_prompts)
    start_index = rank * (total_prompts // torch.cuda.device_count())
    end_index = (rank + 1) * (total_prompts // torch.cuda.device_count()) if rank != torch.cuda.device_count() - 1 else total_prompts

    for name in tqdm(sample_prompts[start_index:end_index], desc=f'pid-[{str(rank)}]'):
        
        value = data[name]

        target_im_pil = None
        
        image_path = os.path.join(dir_path, value["id"])
        prompt = value["prompt"]

        tmp_template = random.choice(english_img2img_template)
        
        source_img_pil = Image.open(image_path).convert('RGB')

        line = prompt

        inputs, tgt_h, tgt_w = prepare_edit_infer_inputs(model, pixel_vae_model, processor, config, source_img_pil, line, tmp_template, device)

        null_inputs, _, _ = prepare_edit_infer_inputs(model, pixel_vae_model, processor, config, Image.new('RGB', source_img_pil.size, (0, 0, 0)), '', tmp_template, device)

        # print("finish prepare input and null_input")

        gpt_rsp_template = "<|generate_start|>"
        gpt_out = processor.tokenizer(gpt_rsp_template)
        
        inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor(gpt_out['input_ids'], device=device).unsqueeze(0)), dim=1)
        inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.tensor(gpt_out['attention_mask'], device=device).unsqueeze(0)), dim=1)

        null_inputs['input_ids'] = torch.cat((null_inputs['input_ids'], torch.tensor(gpt_out['input_ids'], device=device).unsqueeze(0)), dim=1)
        text_tokens_and_mask = processor.tokenizer.pad(
                                {'input_ids': [null_inputs['input_ids'].squeeze(0)]},
                                max_length=inputs['input_ids'].shape[1],
                                padding='max_length',
                                # truncation=True,
                                return_attention_mask=True,
                                padding_side='left',
                                return_tensors='pt').to(device)

        null_inputs.update(text_tokens_and_mask)

        try:
            generated_ids = model.mar_i2i_generate(inputs, null_inputs, vqvae_model, eot_token=tokenizer.eos_token_id, top_k=1000, top_p=1.0, cfg=cfg_scale, generate_start_id=generate_start_id, generate_end_id=generate_end_id, eol_id=eol_id, tgt_h=tgt_h, tgt_w=tgt_w, ds=16).squeeze(0)
            
            output = vqvae_model.decode_code(generated_ids, torch.Size([1, 8, tgt_h // 16, tgt_w // 16])) # output value is between [-1, 1]

            # postprocess
            output = F.interpolate(output, size=[tgt_h, tgt_w], mode='bicubic').permute(0, 2, 3, 1)[0]
            sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

            # save        
            if target_im_pil is None:
                
                target_im_pil = source_img_pil
            
            Image.fromarray(sample).save(f"{edited_path}/{name}.png")
            
            horizontal_concat_3images(source_img_pil, Image.fromarray(sample), target_im_pil).save(f"{output_path}/{line[:50]}.jpg")
        
        except Exception as e:
            print(e)
            pass


from PIL import Image

def horizontal_concat_3images(source_img_pil, sample, target_img_pil, tgt_h=384, tgt_w=384):
    """

    param：
        source_img_pil : PIL.Image - left image
        sample         : PIL.Image - middle image
        target_img_pil : PIL.Image - right image
        tgt_h          : int       - target height
        tgt_w          : int       - target width

    return：
        PIL.Image - concat (3*tgt_w, tgt_h) image
    """
    # resize to target size
    imgs = [
        source_img_pil.resize((tgt_w, tgt_h), Image.BICUBIC),
        sample.resize((tgt_w, tgt_h), Image.BICUBIC),
        target_img_pil.resize((tgt_w, tgt_h), Image.BICUBIC)
    ]

    new_img = Image.new('RGB', (tgt_w * 3, tgt_h))
    for idx, img in enumerate(imgs):
        new_img.paste(img, (idx * tgt_w, 0))
    return new_img


# example
# new_img = horizontal_concat_3images(img1, img2, img3, tgt_h=256, tgt_w=384)


if __name__ == '__main__':
    weight_dtype = torch.bfloat16

    model = Wallaroo(**config.model.wallaroo).to(device).to(weight_dtype)
    model.eval()

    model.wallaroo.config.pixel_start_token_id = pixel_start_id
    model.wallaroo.config.pixel_end_token_id = pixel_end_id
    model.wallaroo.config.generate_start_token_id = generate_start_id
    model.wallaroo.config.generate_end_token_id = generate_end_id
    model.wallaroo.config.generate_img_pad_id = image_generate_pad_id
    model.wallaroo.config.eol_id = eol_id

    resume_checkpoint = torch.load(config.pretrained_path, map_location="cpu")
    m, u = model.load_state_dict(resume_checkpoint['state_dict'], strict=False)
    print(f"transformer model missing keys: {len(m)}, unexpected keys: {len(u)}")
    assert len(u) == 0, f"Found unexpected keys: {u}, please check the ckpt carefully."
    del resume_checkpoint

    w_pixel_vae = config.model.wallaroo.w_pixel_vae
    pixel_pad_left = config.model.wallaroo.pixel_pad_left
    model.wallaroo.config.w_pixel_vae = w_pixel_vae
    model.wallaroo.config.pixel_pad_left = pixel_pad_left

    # vqvae model for t2i image-encoding
    vqvae_config = config.model.vqvae_model
    vqvae_model = build_image_tokenizer(vqvae_config)
    vqvae_model.eval()
    vqvae_model.requires_grad_(False)

    vae_weight_dtype = torch.float32
    
    if w_pixel_vae:
        pixel_vae_model = vqvae_model
        pixel_vae_model.eval()
        pixel_vae_model.requires_grad_(False)

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    wallaroo_config = config.model.wallaroo

    cfg_scale = config.cfg
    tgt_w = tgt_h = config.resolution

    output_path = f"{config.save_path}_{cfg_scale}"
    if not os.path.exists(output_path): os.makedirs(output_path)

    image_token_count = config.image_token_count

    from data_curation import TFReader, MultiTFReader
    from io import BytesIO

    # path = "/datasets/GEdit-Bench/tfrecords"

    # path = "/edit_dataset/UltraEdit_CN_Enrich/"

    path = "/zhujie54/Benchmark/singleturn/singleturn.json"

    with open(path, 'r') as file:
        data = json.load(file)

    dir_path = "/zhujie54/Benchmark/singleturn"

    edited_path = output_path.replace("generate_output", "edit_bench_output")

    if not os.path.exists(edited_path): os.makedirs(edited_path)
    
    exit_files = os.listdir(edited_path)

    for name in exit_files:

        base = name.split('.')[0]
        data.pop(base)

    keys = list(data.keys())

    print("exist file number", len(keys))

    cfg_scale = config.cfg
    image_token_count = config.image_token_count
    world_size = torch.cuda.device_count()

    # Multi-GPU Inference with torch.multiprocessing.spawn
    torch.multiprocessing.spawn(run_inference_on_gpu, args=(model, processor, pixel_vae_model, vqvae_model, keys, data, dir_path, output_path, edited_path, cfg_scale, image_token_count), nprocs=world_size)


        
