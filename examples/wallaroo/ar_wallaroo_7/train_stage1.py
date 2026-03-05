# coding=utf-8
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

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
import numpy as np
import wandb
import torch
import glob
import gc

from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
from mmcv.runner import LogBuffer
from torch.optim import AdamW
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType, set_seed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from wallaroo.models import Wallaroo
from wallaroo.models.tokenizer_image import build_image_tokenizer
from wallaroo.datasets import ImageNetDataset
from wallaroo.datasets.constants import *
from wallaroo.optimizers import *
from wallaroo.utils.simple_utils import get_config, flatten_omega_conf
from wallaroo.utils.logging import set_verbosity_info, set_verbosity_error
from wallaroo.utils.vis_utils import visualize_mmu_predictions, visualize_t2i_predictions_1d

from accelerate.logging import get_logger
logger = get_logger('Wallaroo', log_level="INFO")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

try:
    import apex
    is_apex_available = True
except ImportError:
    is_apex_available = False


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'Qwen2DecoderLayer'
    os.environ["FSDP_USE_ORIG_PARAMS"] = 'true'
    os.environ["FSDP_FORWARD_PREFETCH"] = 'true'
    os.environ["FSDP_SHARDING_STRATEGY"] = 'FULL_SHARD'


def process_t2i_batch(t2i_batch, device):
    batch_size_t2i = t2i_batch["input_ids"].shape[0]
    t2i_image_grid_thw = t2i_batch['image_grid_thw'].to(device, non_blocking=True)
    t2i_pixel_values, t2i_input_ids, t2i_attention_mask, t2i_labels = t2i_batch["images"], t2i_batch["input_ids"].to(accelerator.device, non_blocking=True), t2i_batch["attention_mask"].to(accelerator.device, non_blocking=True), t2i_batch["labels"].to(accelerator.device, non_blocking=True)

    modified_input_ids, modified_text_token_labels, modified_image_token_labels = torch.clone(t2i_input_ids), torch.clone(t2i_labels), torch.clone(t2i_labels)
    continue_vq_features = [] 

    for i, (t2i_input_id, t2i_label, t2i_pixel_value) in enumerate(zip(t2i_input_ids, t2i_labels, t2i_pixel_values)):
        with torch.no_grad():
            if  vqvae_config.type in ['LlamaGen_16x16',  'MOVQGAN_8x8']:
                z_q, _, [_, _, t2i_indice] = vqvae_model.encode(t2i_pixel_value.to(accelerator.device, non_blocking=True).unsqueeze(0))
                codebook_entry = vqvae_model.quantize.get_codebook_entry(t2i_indice).unsqueeze(0).to(device).to(weight_dtype)
            else:
                print("Not support vqvae type")
                raise NotImplementedError
        
        continue_vq_feature = model.gen_input_adpater(codebook_entry)
        continue_vq_features.append(continue_vq_feature)

        generate_image_mask = (t2i_input_id == image_generate_pad_id)
        generate_img_tokens = generate_image_mask.nonzero(as_tuple=True)[0].shape[0]
        if generate_img_tokens != t2i_indice.shape[0]:
            raise ValueError(
                            f"T2i image pad tokens and vqvae tokens do not match: tokens: {generate_img_tokens}, features {t2i_indice.shape[0]}"
                        )

        modified_input_ids[i] = t2i_input_id
        modified_image_token_labels[i] = torch.full(t2i_input_id.shape, -100).to(device).masked_scatter(generate_image_mask, t2i_indice)
        modified_text_token_labels[i] = t2i_label.masked_fill(generate_image_mask, -100)
    
    t2i_attention_mask = t2i_attention_mask.to(mask_dtype)
    continue_vq_features = torch.cat(continue_vq_features, dim=0) 

    return modified_input_ids, continue_vq_features, t2i_attention_mask, modified_image_token_labels, modified_text_token_labels, t2i_image_grid_thw
    
def train(global_step=0):

    model.train()

    time_start, last_tic = time.time(), time.time()
    
    for epoch in range(first_epoch, num_train_epochs):
        data_time_start= time.time()
        data_time_all = 0

        for step, batch in enumerate(train_dataloader_t2i):

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            t2i_input_ids, continue_vq_features, t2i_attention_mask, image_token_labels, text_token_labels, t2i_image_grid_thw = process_t2i_batch(t2i_batch=batch, device=accelerator.device)

            t2i_mm_inputs = {'input_ids': t2i_input_ids, 
                             'continue_vq_features':continue_vq_features, 
                             'attention_mask': t2i_attention_mask, 
                             'image_token_labels': image_token_labels, 
                             'text_token_labels':text_token_labels}

            grad_norm = None 

            with accelerator.accumulate(model):
                
                t2i_loss, text_token_loss, image_token_loss = model.forward_t2i(t2i_inputs=t2i_mm_inputs)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_image_token_loss = accelerator.gather(image_token_loss.repeat(config.training.batch_size_t2i)).mean()
                avg_text_token_loss = accelerator.gather(text_token_loss.repeat(config.training.batch_size_t2i)).mean()
                avg_t2i_loss = accelerator.gather(t2i_loss.repeat(config.training.batch_size_t2i)).mean()

                loss = config.training.t2i_coeff * t2i_loss
                accelerator.backward(loss)
                optimizer.step()

                # mixed-precision training, precision overflow causes skipped, without updating the learning rate
                if not accelerator.optimizer_step_was_skipped: 
                    lr_scheduler.step()
                
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    grad_norm = model.get_global_grad_norm()

            if accelerator.sync_gradients:

                lr = lr_scheduler.get_last_lr()[0]
                logs = {'loss': accelerator.gather(loss).mean().item(), 't2i_loss': avg_t2i_loss.item(), 'text_token_loss': avg_text_token_loss.item(), 'image_token_loss': avg_image_token_loss.item()}
                
                if grad_norm is not None:
                    logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())

                log_buffer.update(logs)

                if (global_step + 1) % config.experiment.log_every == 0 or (global_step + 1) == 1:
                    t = (time.time() - last_tic) / config.experiment.log_every
                    t_d = data_time_all / config.experiment.log_every
                    avg_time = (time.time() - time_start) / (global_step + 1)

                    log_buffer.average()
                    info = f"Step={step+1}, Epoch={epoch}, global_step={global_step+1}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "
                    info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                    logger.info(info)
                    last_tic = time.time()
                    log_buffer.clear()
                    data_time_all = 0
                
                logs.update(lr=lr)
                accelerator.log(logs, step=global_step)

                global_step += 1
                data_time_start= time.time()
                
                # Save model checkpoint
                if (global_step) % config.experiment.save_every == 0:
                    save_path = os.path.join(config.experiment.output_dir, 'checkpoints')
                    torch.cuda.empty_cache()
                    model.save_checkpoint(save_path, tag=f"checkpoint-{global_step}-ds", save_latest=False)

                    if accelerator.is_main_process:
                        state_dict_fp32 = get_fp32_state_dict_from_zero_checkpoint(save_path, tag=f"checkpoint-{global_step}-ds")
                        state_dict = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "state_dict": state_dict_fp32 #to save full-precesion model
                        }

                        last_checkpoint_path = os.path.join(save_path, f"checkpoint-{global_step}.ckpt")
                        torch.save(state_dict, last_checkpoint_path)
                        # os.system(f"rm -rf {os.path.join(save_path, f'checkpoint-{global_step}-ds')}")
                        os.system(f"rm -rf {os.path.join(save_path, 'zero_to_fp32.py')}")
                        logger.info(f"Saved state to {save_path} (global_step: {global_step})")

                    accelerator.wait_for_everyone()
                    
                if (global_step) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    logger.info("Visualizing mmu predictions...")
                    visualize_mmu_predictions(model, processor, config, global_step+1, accelerator.device)
                    logger.info("Visualizing t2i predictions...")
                    visualize_t2i_predictions_1d(model, vqvae_model, processor, config, global_step+1, accelerator.device, image_token_count=(preproc_config.t2i_resolution // preproc_config.vqvae_downscale) ** 2, max_new_tokens=2048, target_resolution=preproc_config.t2i_resolution)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    config = get_config()
    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")

    os.umask(0o000)
    config.work_dir = config.experiment.output_dir
    config.load_from = config.experiment.load_from
    if config.load_from == 'None': config.load_from = None
    os.makedirs(config.work_dir, exist_ok=True)

    # Perform auto-resume
    auto_resume_success_pt = False
    auto_resume_success_ds = False
    resume_checkpoint_path = None 
    resume_ds_state_path = None

    if config.auto_resume:
        checkpoints_dir = os.path.join(config.work_dir, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            print(f'FOUND checkpoints_dir: {checkpoints_dir}')
            resume_ckpts = os.listdir(checkpoints_dir)
            resume_ckpts_pt = [name for name in resume_ckpts if "ds" not in name]
            resume_ckpts_ds = [name for name in resume_ckpts if "ds" in name]

            resume_ckpts_pt = sorted(resume_ckpts_pt, key=lambda x:-int(x.split(".")[0].split("-")[1]))
            resume_ckpts_ds = sorted(resume_ckpts_ds, key=lambda x:-int(x.split("-")[1]))

            print(f'resume_ckpts_pt: {resume_ckpts_pt}')
            print(f'resume_ckpts_ds: {resume_ckpts_ds}')

            if resume_ckpts_pt: 
                resume_ckpt_name_pt = resume_ckpts_pt[0]
                print(f'FOUND resume_ckpt_name: {resume_ckpt_name_pt}')
                auto_resume_success_pt = True
            
            if resume_ckpts_ds: 
                resume_ckpt_name_ds = resume_ckpts_ds[0]
                print(f'FOUND resume_ckpt_name: {resume_ckpt_name_ds}')
                auto_resume_success_ds = True

    # Replace resume ckpt if auto-resume triggers
    if auto_resume_success_pt: 
        # If auto_resume_success, resume_global_step should be true to keep step tracking
        resume_global_step = True 
        resume_checkpoint_path = os.path.join(checkpoints_dir, resume_ckpt_name_pt)
        print(f"Auto_resume_pt_success! Resume ckpt is set as {resume_checkpoint_path}.")
        config.load_from = None
    else:
        resume_checkpoint_path = None  # Ensure an explicit assignment when a checkpoint is not found
    
    if auto_resume_success_ds: 
        resume_ds_state_path = os.path.join(checkpoints_dir, resume_ckpt_name_ds)
        print(f"Auto_resume_ds_success! Resume ckpt is set as {resume_ds_state_path}.")


    even_batches = True
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        bucket_cap_mb = config.get('bucket_cap_mb', 300)
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
        ddp_kwargs = None
    else:
        init_train = 'DDP'
        fsdp_plugin = None
        bucket_cap_mb = config.get('bucket_cap_mb', 300)
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, bucket_cap_mb=bucket_cap_mb)

    deepspeed_plugin = None
    
    total_batch_size_per_gpu = config.training.batch_size_t2i 


    if config.use_deepspeed:
        init_train = 'DeepSpeed'
        from accelerate import DeepSpeedPlugin
        deepspeed_config = json.load(open(config.deepspeed_config, 'r'))


    even_batches = True
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        bucket_cap_mb = config.get('bucket_cap_mb', 300)
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
        ddp_kwargs = None
    else:
        init_train = 'DDP'
        fsdp_plugin = None
        bucket_cap_mb = config.get('bucket_cap_mb', 300)
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, bucket_cap_mb=bucket_cap_mb)

    deepspeed_plugin = None
    total_batch_size_per_gpu = config.training.batch_size_t2i

    if config.use_deepspeed:
        init_train = 'DeepSpeed'
        from accelerate import DeepSpeedPlugin
        deepspeed_config = json.load(open(config.deepspeed_config, 'r'))
        deepspeed_config['train_micro_batch_size_per_gpu'] = total_batch_size_per_gpu
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config, gradient_clipping=config.training.max_grad_norm)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
        project_dir=config.experiment.logging_dir,
        kwargs_handlers=[init_handler] if ddp_kwargs is None else [init_handler, ddp_kwargs]
    )

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    total_batch_size = ((config.training.batch_size_t2i)* accelerator.num_processes * config.training.gradient_accumulation_steps)

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (total_batch_size_per_gpu)
    
    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")
        
        wandb.login(key=config.wandb.get('key', 'test_exp'))
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        seed = config.training.seed
        set_seed(seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    processor = AutoProcessor.from_pretrained(config.model.wallaroo.llm_model_path, cache_dir=config.model.wallaroo.cache_dir)
    tokenizer = processor.tokenizer

    if 'Qwen' in config.model.wallaroo.llm_model_path:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    special_tokens = (GENERATE_START_TOKEN, GENERATE_END_TOKEN, PIXEL_START_TOKEN, PIXEL_END_TOKEN, EOL_TOKEN, DEFAULT_GENERATE_IMAGE_TOKEN)

    #tokenizer.add_tokens(list(special_tokens))
    tokenizer.add_tokens(list(special_tokens) + hw_indicator_256_lst)

    generate_start_id = processor.tokenizer.convert_tokens_to_ids(GENERATE_START_TOKEN)
    generate_end_id = processor.tokenizer.convert_tokens_to_ids(GENERATE_END_TOKEN)
    pixel_start_id = processor.tokenizer.convert_tokens_to_ids(PIXEL_START_TOKEN)
    pixel_end_id = processor.tokenizer.convert_tokens_to_ids(PIXEL_END_TOKEN)
    eol_id = processor.tokenizer.convert_tokens_to_ids(EOL_TOKEN)
    image_pad_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    image_generate_pad_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_GENERATE_IMAGE_TOKEN)

    # vqvae model for t2i image-encoding
    vqvae_config = config.model.vqvae_model
    vqvae_model = build_image_tokenizer(vqvae_config)
    vqvae_model.eval()
    vqvae_model.requires_grad_(False)

    # Initialize wallaroo model
    model = Wallaroo(**config.model.wallaroo).to(accelerator.device)
    model.wallaroo.config.pixel_start_token_id = pixel_start_id
    model.wallaroo.config.pixel_end_token_id = pixel_end_id
    model.wallaroo.config.generate_start_token_id = generate_start_id
    model.wallaroo.config.generate_end_token_id = generate_end_id
    model.wallaroo.config.generate_img_pad_id = image_generate_pad_id

    semantic_vision_tower = model.wallaroo.visual.to(accelerator.device)
    semantic_vision_tower.eval()
    for p in semantic_vision_tower.parameters():
        p.requires_grad = False

    ## fix all parameter in Qwen VL model
    for name, param in model.named_parameters():
        if "gen_image_head" in name or "gen_input_adpater" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if config.load_from is not None:
        resume_checkpoint = torch.load(config.load_from, map_location="cpu")
        m, u = model.load_state_dict(resume_checkpoint['state_dict'], strict=False)
        logger.info(f"load from checkpoint: {config.load_from}")
        logger.info(f"transformer model missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0, f"Found unexpected keys: {u}, please check the ckpt carefully."
        del resume_checkpoint
        gc.collect()

    ##################################
    #   resume ckpt   #
    #################################

    first_epoch, global_step = 0, 0
    if resume_checkpoint_path:
        assert os.path.exists(resume_checkpoint_path), f"Error: {resume_checkpoint_path} NOT EXIST"
        logger.info(f"resume from checkpoint: {resume_checkpoint_path}")
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
        # resume dit parameters
        print(f'resume_checkpoint keys: {resume_checkpoint.keys()}')

        assert "state_dict" in resume_checkpoint, f"Resume_checkpoint {resume_checkpoint_path} should contain valid state_dict."
        state_dict = resume_checkpoint["state_dict"]

        m, u = model.load_state_dict(state_dict, strict=False)
        logger.info(f"transformer model missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0, f"Found unexpected keys: {u}, please check the ckpt carefully."

        # resume global step
        if "global_step" in resume_checkpoint and resume_global_step:
            logger.info(f"resume global_step: {resume_checkpoint['global_step']}")
            global_step = resume_checkpoint['global_step']       
            seed = seed + global_step
            set_seed(seed)
            logger.info(f'Add to seed by global step --> seed: {seed}')
        
        del resume_checkpoint
        del state_dict
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.empty_cache()

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )
    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i = (config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps)

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    wallaroo_config = config.model.wallaroo
    
    if config.dataset.gen_type == "imagenet":
        dataset_imagenet = ImageNetDataset(
            dataset_config.train_t2i_shards_path_or_url,
            image_size=preproc_config.t2i_resolution,
            processor=processor,
            uncond_p=preproc_config.uncond_p,
            vqvae_downscale=preproc_config.vqvae_downscale,
            t2i_generate_method=config.model.wallaroo.t2i_generate_method,
            hw_indicator=preproc_config.hw_indicator,
            max_seq_length=preproc_config.t2i_max_seq_length
        )

        if accelerator.num_processes > 1:
            sampler = DistributedSampler(dataset_imagenet,
                                         num_replicas=accelerator.num_processes,
                                         rank=accelerator.process_index,
                                         shuffle=True,)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_dataloader_t2i = DataLoader(dataset_imagenet, batch_size=config.training.batch_size_t2i,
                                          sampler=sampler, collate_fn=dataset_imagenet.collate_fn,
                                          shuffle=shuffle, num_workers=dataset_config.num_workers)

        num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    else:
        raise ValueError(f"Unsupported dataset type {config.dataset.type}")


    ##################################
    #       Prepare accelerator     #
    #################################
    vqvae_model.to(device=accelerator.device)

    if hasattr(model, 'module'):
        mask_dtype = model.module.wallaroo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.wallaroo.model.embed_tokens.weight.dtype

    logger.info("Preparing model, optimizer and dataloaders")
    model, _, _ = accelerator.prepare(model, optimizer, lr_scheduler)

    if resume_ds_state_path:
        logger.info(f"Resume DS state from {resume_ds_state_path}...")
        model.load_checkpoint(os.path.dirname(resume_ds_state_path), tag=os.path.basename(resume_ds_state_path))

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    log_buffer = LogBuffer()
    train(global_step=global_step)

