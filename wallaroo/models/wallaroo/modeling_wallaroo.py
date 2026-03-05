# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoConfig, AutoModel
from wallaroo.models.common.modeling_utils import ConfigMixin, ModelMixin, register_to_config
from wallaroo.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPreTrainedModel, Qwen2_5_VLForConditionalGeneration


class Wallaroo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_pixel_vae,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            codebook_embed_dim=8,
            num_vq_tokens=256,
            patch_size=2,
            in_channels=16,
            hidden_size=3584,
            load_from_wallaroo=True,
            cache_dir=None,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.w_pixel_vae = w_pixel_vae
        self.llm_vocab_size = llm_vocab_size


        self.wallaroo = Qwen2_5_VLForConditionalGeneration.from_pretrained(llm_model_path, cache_dir=cache_dir, trust_remote_code=True, attn_implementation='sdpa')
        

        self.codebook_size = codebook_size
        self.output_size = self.vocab_size

        if w_pixel_vae:
            self.pixel_adapter = torch.nn.Sequential(nn.Linear(codebook_embed_dim, hidden_size),
                                                        nn.GELU(),
                                                        nn.Linear(hidden_size, hidden_size))
        
            self.edit_image_head = torch.nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                        nn.GELU(),
                                                        nn.Linear(hidden_size, codebook_size, bias=False))

            self.edit_input_adpater = torch.nn.Sequential(nn.Linear(codebook_embed_dim, hidden_size),
                                                        nn.GELU(),
                                                        nn.Linear(hidden_size, hidden_size))

        self.gen_image_head = torch.nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                        nn.GELU(),
                                                        nn.Linear(hidden_size, codebook_size, bias=False))
                                                    
        self.gen_input_adpater = torch.nn.Sequential(nn.Linear(codebook_embed_dim, hidden_size),
                                                        nn.GELU(),
                                                        nn.Linear(hidden_size, hidden_size))
        
        self.initialize_weights()

    def initialize_weights(self):
        pass

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
    
    def cal_t2i_loss(self, input_ids, text_t2i_labels, image_t2i_labels, t2i_output):
        hidden_states = t2i_output.hidden_states[-1]
        text_logits = t2i_output.logits.float()
        assert torch.equal(text_logits, self.wallaroo.lm_head(hidden_states)) == True
        img_logits = self.gen_image_head(hidden_states).float()

        # Shift so that tokens < n predict n
        shift_img_logits = img_logits[..., :-1, :].contiguous()
        shift_text_logits = text_logits[..., :-1, :].contiguous()
        shift_text_labels = text_t2i_labels[..., 1:].contiguous()
        shift_image_labels = image_t2i_labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_text_logits = shift_text_logits.view(-1, self.config.vocab_size)
        shift_img_logits = shift_img_logits.view(-1, self.codebook_size)

        shift_text_labels =  shift_text_labels.view(-1)
        shift_image_labels =  shift_image_labels.view(-1)
        # Enable model parallelism
        shift_text_labels =  shift_text_labels.to(shift_text_logits.device)
        shift_image_labels =  shift_image_labels.to(shift_img_logits.device)
        text_token_loss = loss_fct(shift_text_logits, shift_text_labels)
        image_token_loss = loss_fct(shift_img_logits, shift_image_labels)

        t2i_output.loss = text_token_loss + image_token_loss

        return t2i_output.loss, text_token_loss, image_token_loss

    
    def cal_i2i_loss(self, input_ids, text_t2i_labels, image_t2i_labels, t2i_output):
        hidden_states = t2i_output.hidden_states[-1]
        text_logits = t2i_output.logits.float()
        assert torch.equal(text_logits, self.wallaroo.lm_head(hidden_states)) == True
        img_logits = self.edit_image_head(hidden_states).float()

        # Shift so that tokens < n predict n
        shift_img_logits = img_logits[..., :-1, :].contiguous()
        shift_text_logits = text_logits[..., :-1, :].contiguous()
        shift_text_labels = text_t2i_labels[..., 1:].contiguous()
        shift_image_labels = image_t2i_labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_text_logits = shift_text_logits.view(-1, self.config.vocab_size)
        shift_img_logits = shift_img_logits.view(-1, self.codebook_size)

        shift_text_labels =  shift_text_labels.view(-1)
        shift_image_labels =  shift_image_labels.view(-1)
        # Enable model parallelism
        shift_text_labels =  shift_text_labels.to(shift_text_logits.device)
        shift_image_labels =  shift_image_labels.to(shift_img_logits.device)
        text_token_loss = loss_fct(shift_text_logits, shift_text_labels)
        image_token_loss = loss_fct(shift_img_logits, shift_image_labels)

        t2i_output.loss = text_token_loss + image_token_loss

        return t2i_output.loss, text_token_loss, image_token_loss

    def forward_t2i(self, t2i_inputs):
        text_t2i_labels = t2i_inputs.pop("text_token_labels")
        image_t2i_labels = t2i_inputs.pop("image_token_labels")
        
        t2i_inputs['output_hidden_states'] = True
        t2i_output = self.wallaroo(**t2i_inputs)

        t2i_loss, text_token_loss, image_token_loss = self.cal_t2i_loss(t2i_inputs['input_ids'], text_t2i_labels, image_t2i_labels, t2i_output)
        
        return t2i_loss, text_token_loss, image_token_loss

    
    def forward_i2i(self, t2i_inputs):
        text_t2i_labels = t2i_inputs.pop("text_token_labels")
        image_t2i_labels = t2i_inputs.pop("image_token_labels")
        
        t2i_inputs['output_hidden_states'] = True
        t2i_output = self.wallaroo(**t2i_inputs)

        t2i_loss, text_token_loss, image_token_loss = self.cal_i2i_loss(t2i_inputs['input_ids'], text_t2i_labels, image_t2i_labels, t2i_output)
        
        return t2i_loss, text_token_loss, image_token_loss

    def forward_mmu(self, mmu_inputs,
            **kwargs):
        
        mmu_output = self.wallaroo(**mmu_inputs)
        
        return mmu_output.loss

    def forward(
            self,
            mmu_inputs,
            t2i_inputs,
            **kwargs,
    ):  
        t2i_loss, text_token_loss, image_token_loss = self.forward_t2i(t2i_inputs)
        mmu_output = self.wallaroo(**mmu_inputs)
        
        return mmu_output.loss, t2i_loss, text_token_loss, image_token_loss

    def forward_omni(
            self,
            mmu_inputs,
            t2i_inputs,
            edit_inputs,
            **kwargs,
    ):  
        
        mmu_output = self.wallaroo(**mmu_inputs)
        t2i_loss, t2i_text_token_loss, t2i_image_token_loss = self.forward_t2i(t2i_inputs)
        edit_loss, edit_text_token_loss, edit_image_token_loss = self.forward_i2i(edit_inputs)

        return mmu_output.loss, t2i_loss, t2i_text_token_loss, t2i_image_token_loss, edit_loss, edit_text_token_loss, edit_image_token_loss
    
    def mmu_generate(self, mm_inputs, eot_token, max_new_tokens=256, temperature=1.0, top_k=1, top_p=0.8):
        result = []
        for _ in range(max_new_tokens):
            logits = self.wallaroo(**mm_inputs).logits
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            result.append(idx_next)
            
            # append
            mm_inputs['input_ids'] = torch.cat((mm_inputs['input_ids'], idx_next), dim=1)
            mm_inputs['attention_mask'] = torch.cat((mm_inputs['attention_mask'], torch.ones((1, 1)).to(mm_inputs['input_ids'].device)), dim=1)

            
            if eot_token is not None and idx_next.cpu() == eot_token:
                break
        
        return torch.cat(result, dim=1)
    ### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
    
    def top_k_top_p_filtering(
        self,
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits


    def sample(self, logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
        logits = logits / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample_logits:
            idx = torch.multinomial(probs, num_samples=1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
        return idx, probs

    
    @torch.no_grad()
    def t2i_generate(self, t2i_inputs, null_t2i_inputs, vqvae_model, eot_token, cfg=1.0, max_new_tokens=512, temperature=1.0, top_k=2000, top_p=1.0, generate_start_id=None, generate_end_id=None, sep_token_id=151643, img_token_count=576, tgt_h=384, tgt_w=384, ds=8):
        result = []

        B, device = t2i_inputs['input_ids'].shape[0], t2i_inputs['input_ids'].device    

        assert t2i_inputs['input_ids'].shape[1] == null_t2i_inputs['input_ids'].shape[1]
        assert t2i_inputs['attention_mask'].shape[1] == null_t2i_inputs['attention_mask'].shape[1]

        # get preprared position id of t2i_inputs
        fake_input_ids = torch.cat([t2i_inputs['input_ids'], torch.ones([B, img_token_count]).long().to(device) * self.wallaroo.config.generate_img_pad_id], dim=1)
        fake_attention_mask = torch.cat([t2i_inputs['attention_mask'], torch.ones([B, img_token_count]).to(device)], dim=1)
        position_ids, _ = self.wallaroo.get_rope_index(input_ids=fake_input_ids, attention_mask=fake_attention_mask, image_grid_thw=None)

        # get preprared position id of null_t2i_inputs 
        fake_null_input_ids = torch.cat([null_t2i_inputs['input_ids'], torch.ones([B, img_token_count]).long().to(device) * self.wallaroo.config.generate_img_pad_id], dim=1)
        fake_null_attention_mask = torch.cat([null_t2i_inputs['attention_mask'], torch.ones([B, img_token_count]).to(device)], dim=1)
        null_position_ids, _ = self.wallaroo.get_rope_index(input_ids=fake_null_input_ids, attention_mask=fake_null_attention_mask, image_grid_thw=None)

        if cfg > 1.0:
            t2i_input_embeds = self.wallaroo.model.embed_tokens(torch.cat([t2i_inputs['input_ids'], null_t2i_inputs['input_ids']], dim=0))
            t2i_attention_mask = torch.cat([t2i_inputs['attention_mask'], null_t2i_inputs['attention_mask']], dim=0)
            t2i_position_ids = torch.cat([position_ids[:, :, :t2i_inputs['input_ids'].shape[1]], null_position_ids[:, :, :null_t2i_inputs['input_ids'].shape[1]]], dim=1)
        else:
            t2i_input_embeds = self.wallaroo.model.embed_tokens(t2i_inputs['input_ids'])
            t2i_attention_mask = t2i_inputs['attention_mask']
            t2i_position_ids = position_ids[:, :, :t2i_attention_mask.shape[1]]
        
        for i in range(img_token_count):
            outputs = self.wallaroo(inputs_embeds=t2i_input_embeds, attention_mask=t2i_attention_mask, position_ids=t2i_position_ids, output_hidden_states=True, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.hidden_states[-1]
            logits = self.gen_image_head(hidden_states)
            if cfg > 1.0:
                logits, null_logits = torch.chunk(logits, 2, dim=0)
                cfg_logits = null_logits[:, -1, :] + cfg * (logits[:, -1, :] - null_logits[:, -1, :])
            else:
                cfg_logits = logits[:, -1, :]

            idx_next, _ = self.sample(cfg_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            result.append(idx_next)
            
            t2i_input_embeds = self.gen_input_adpater(vqvae_model.quantize.get_codebook_entry(idx_next).to(torch.bfloat16))
            t2i_attention_mask = torch.ones((B, 1)).to(t2i_inputs['input_ids'].device)
            t2i_position_ids = position_ids[:, :, t2i_inputs['input_ids'].shape[1]+i:t2i_inputs['input_ids'].shape[1]+i+1]
            
            if cfg > 1.0:
                t2i_input_embeds = t2i_input_embeds.repeat(2, 1, 1)
                t2i_attention_mask = t2i_attention_mask.repeat(2, 1)
                null_t2i_position_ids = null_position_ids[:, :, t2i_inputs['input_ids'].shape[1]+i:t2i_inputs['input_ids'].shape[1]+i+1]  ## different, can not repeat
                t2i_position_ids = torch.cat([t2i_position_ids, null_t2i_position_ids], dim=1)

        return torch.cat(result, dim=1)

    @torch.no_grad()
    def mar_t2i_generate(self, t2i_inputs, null_t2i_inputs, vqvae_model, eot_token, cfg=1.0, temperature=1.0, top_k=2000, top_p=1.0, generate_start_id=None, generate_end_id=None, eol_id=151643, tgt_h=384, tgt_w=384, ds=8):
        
        result = []
        continue_features = [] 

        img_token_count = (tgt_w//ds) * (tgt_h//ds)
        token_per_row = tgt_w//ds
        end_of_line_num = tgt_h//ds-1

        generate_img_pad_id = self.wallaroo.config.generate_img_pad_id
        B, device = t2i_inputs['input_ids'].shape[0], t2i_inputs['input_ids'].device

        assert t2i_inputs['input_ids'].shape[1] == null_t2i_inputs['input_ids'].shape[1]
        assert t2i_inputs['attention_mask'].shape[1] == null_t2i_inputs['attention_mask'].shape[1]

        # get preprared position id of t2i_inputs
        fake_input_ids = torch.cat([t2i_inputs['input_ids'], torch.ones([B, img_token_count+ end_of_line_num]).long().to(device) * self.wallaroo.config.generate_img_pad_id], dim=1)
        fake_attention_mask = torch.cat([t2i_inputs['attention_mask'], torch.ones([B, img_token_count+ end_of_line_num]).to(device)], dim=1)
        position_ids, _ = self.wallaroo.get_rope_index(input_ids=fake_input_ids, attention_mask=fake_attention_mask, image_grid_thw=None)

        # get preprared position id of null_t2i_inputs 
        fake_null_input_ids = torch.cat([null_t2i_inputs['input_ids'], torch.ones([B, img_token_count+end_of_line_num]).long().to(device) * self.wallaroo.config.generate_img_pad_id], dim=1)
        fake_null_attention_mask = torch.cat([null_t2i_inputs['attention_mask'], torch.ones([B, img_token_count+end_of_line_num]).to(device)], dim=1)
        null_position_ids, _ = self.wallaroo.get_rope_index(input_ids=fake_null_input_ids, attention_mask=fake_null_attention_mask, image_grid_thw=None)

        if cfg > 1.0:
            t2i_input_embeds = self.wallaroo.model.embed_tokens(torch.cat([t2i_inputs['input_ids'], null_t2i_inputs['input_ids']], dim=0))
            t2i_attention_mask = torch.cat([t2i_inputs['attention_mask'], null_t2i_inputs['attention_mask']], dim=0)
            t2i_position_ids = torch.cat([position_ids[:, :, :t2i_inputs['input_ids'].shape[1]], null_position_ids[:, :, :null_t2i_inputs['input_ids'].shape[1]]], dim=1)
        else:
            t2i_input_embeds = self.wallaroo.model.embed_tokens(t2i_inputs['input_ids'])
            t2i_attention_mask = t2i_inputs['attention_mask']
            t2i_position_ids = position_ids[:, :, :t2i_attention_mask.shape[1]]
        
        for i in range(0, img_token_count+end_of_line_num):

            if (i+1)%(token_per_row+1)!=0:

                outputs = self.wallaroo(inputs_embeds=t2i_input_embeds, attention_mask=t2i_attention_mask, position_ids=t2i_position_ids, output_hidden_states=True, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.hidden_states[-1]
                logits = self.gen_image_head(hidden_states)

                if cfg > 1.0:
                    logits, null_logits = torch.chunk(logits, 2, dim=0)
                    cfg_logits = null_logits[:, -1, :] + cfg * (logits[:, -1, :] - null_logits[:, -1, :])
                else:
                    cfg_logits = logits[:, -1, :]

                idx_next, _ = self.sample(cfg_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                result.append(idx_next)

                t2i_input_embeds = self.gen_input_adpater(vqvae_model.quantize.get_codebook_entry(idx_next).to(torch.bfloat16))
                t2i_attention_mask = torch.ones((B, 1)).to(t2i_inputs['input_ids'].device)
                t2i_position_ids = position_ids[:, :, t2i_inputs['input_ids'].shape[1]+i:t2i_inputs['input_ids'].shape[1]+i+1]
                
                if cfg > 1.0:
                    t2i_input_embeds = t2i_input_embeds.repeat(2, 1, 1)
                    t2i_attention_mask = t2i_attention_mask.repeat(2, 1)
                    null_t2i_position_ids = null_position_ids[:, :, t2i_inputs['input_ids'].shape[1]+i:t2i_inputs['input_ids'].shape[1]+i+1]  ## different, can not repeat
                    t2i_position_ids = torch.cat([t2i_position_ids, null_t2i_position_ids], dim=1)

            # add line
            else:
                ## input and save as kv cache
                outputs = self.wallaroo(inputs_embeds=t2i_input_embeds, attention_mask=t2i_attention_mask, position_ids=t2i_position_ids, output_hidden_states=True, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)

                t2i_input_embeds = self.wallaroo.model.embed_tokens(torch.ones((B, 1)).to(t2i_inputs['input_ids'].device).fill_(eol_id).long())
                t2i_attention_mask = torch.ones((B, 1)).to(t2i_inputs['input_ids'].device)
                t2i_position_ids = position_ids[:, :, t2i_inputs['input_ids'].shape[1]+i:t2i_inputs['input_ids'].shape[1]+i+1]
                
                if cfg > 1.0:
                    t2i_input_embeds = t2i_input_embeds.repeat(2, 1, 1)
                    t2i_attention_mask = t2i_attention_mask.repeat(2, 1)
                    null_t2i_position_ids = null_position_ids[:, :, t2i_inputs['input_ids'].shape[1]+i:t2i_inputs['input_ids'].shape[1]+i+1]  ## different, can not repeat
                    t2i_position_ids = torch.cat([t2i_position_ids, null_t2i_position_ids], dim=1)

        return torch.cat(result, dim=1)

    @torch.no_grad()
    def mar_i2i_generate(self, i2i_inputs, null_i2i_inputs, vqvae_model, eot_token, cfg=1.0, temperature=1.0, top_k=2000, top_p=1.0, generate_start_id=None, generate_end_id=None, eol_id=151643, tgt_h=384, tgt_w=384, ds=16):
        
        result = [] 

        B, device = i2i_inputs['input_ids'].shape[0], i2i_inputs['input_ids'].device

        img_token_count = (tgt_w//ds) * (tgt_h//ds)
        token_per_row = tgt_w//ds
        end_of_line_num = tgt_h//ds-1

        assert i2i_inputs['input_ids'].shape[1] == null_i2i_inputs['input_ids'].shape[1]
        assert i2i_inputs['attention_mask'].shape[1] == null_i2i_inputs['attention_mask'].shape[1]

        # get preprared position id of i2i_inputs
        fake_input_ids = torch.cat([i2i_inputs['input_ids'], torch.ones([B, img_token_count+ end_of_line_num]).long().to(device) * self.wallaroo.config.generate_img_pad_id], dim=1)
        fake_attention_mask = torch.cat([i2i_inputs['attention_mask'], torch.ones([B, img_token_count+ end_of_line_num]).to(device)], dim=1)
        position_ids, _ = self.wallaroo.get_rope_index(input_ids=fake_input_ids, attention_mask=fake_attention_mask, image_grid_thw=i2i_inputs['image_grid_thw'])

        # get preprared position id of null_i2i_inputs 
        fake_null_input_ids = torch.cat([null_i2i_inputs['input_ids'], torch.ones([B, img_token_count+end_of_line_num]).long().to(device) * self.wallaroo.config.generate_img_pad_id], dim=1)
        fake_null_attention_mask = torch.cat([null_i2i_inputs['attention_mask'], torch.ones([B, img_token_count+end_of_line_num]).to(device)], dim=1)
        null_position_ids, _ = self.wallaroo.get_rope_index(input_ids=fake_null_input_ids, attention_mask=fake_null_attention_mask, image_grid_thw=null_i2i_inputs['image_grid_thw'])


        if cfg > 1.0:
            input_ids = torch.cat([i2i_inputs['input_ids'], null_i2i_inputs['input_ids']], dim=0)
            t2i_input_embeds = self.wallaroo.model.embed_tokens(input_ids)
            t2i_attention_mask = torch.cat([i2i_inputs['attention_mask'], null_i2i_inputs['attention_mask']], dim=0)
            t2i_position_ids = torch.cat([position_ids[:, :, :i2i_inputs['input_ids'].shape[1]], null_position_ids[:, :, :null_i2i_inputs['input_ids'].shape[1]]], dim=1)
            pixel_values = [i2i_inputs['pixel_values'][0], null_i2i_inputs['pixel_values'][0]]
            pixel_vae_values = [i2i_inputs['pixel_vae_values'][0], null_i2i_inputs['pixel_vae_values'][0]]
            image_grid_thw = torch.cat([i2i_inputs['image_grid_thw'], null_i2i_inputs['image_grid_thw']], dim=0)

        else:
            input_ids = i2i_inputs['input_ids']
            t2i_input_embeds = self.wallaroo.model.embed_tokens(input_ids)
            t2i_attention_mask = i2i_inputs['attention_mask']
            t2i_position_ids = position_ids[:, :, :t2i_attention_mask.shape[1]]
            image_grid_thw = i2i_inputs['image_grid_thw']
            pixel_values = i2i_inputs['pixel_values']
            pixel_vae_values = i2i_inputs['pixel_vae_values']

        new_input_embeds = []

        # add another vae and vit feature 
        for bs_idx, (input_embeds, input_id, pixel_value, image_grid) in enumerate(zip(t2i_input_embeds, input_ids, pixel_values, image_grid_thw)):

            pixel_value = pixel_value.type(self.wallaroo.visual.dtype)
            semantic_image_embed = self.wallaroo.visual(pixel_value, grid_thw=image_grid.unsqueeze(0))

            semantic_start_index = (input_id == self.wallaroo.config.vision_start_token_id).nonzero(as_tuple=True)[0].item()
            semantic_end_index = (input_id == self.wallaroo.config.vision_end_token_id).nonzero(as_tuple=True)[0].item()
            semantic_n_image_tokens = semantic_end_index - semantic_start_index - 1

            semantic_n_image_features = semantic_image_embed.shape[0]

            if semantic_n_image_tokens != semantic_n_image_features:
                raise ValueError(
                    f"Semantic image features and image tokens do not match: tokens: {semantic_n_image_tokens}, features {semantic_n_image_features}"
                )
            
            semantic_image_mask = torch.zeros_like(input_id, dtype=torch.bool)
            semantic_image_mask[semantic_start_index+1:semantic_end_index] = True
            semantic_image_mask = semantic_image_mask.unsqueeze(-1).expand_as(input_embeds).to(input_embeds.device)

            semantic_image_embed = semantic_image_embed.to(t2i_input_embeds.device, t2i_input_embeds.dtype)
            input_embeds = input_embeds.masked_scatter(semantic_image_mask, semantic_image_embed)

            if self.wallaroo.config.w_pixel_vae:

                pixel_start_index = (input_id == self.wallaroo.config.pixel_start_token_id).nonzero(as_tuple=True)[0].item()
                pixel_end_index = (input_id == self.wallaroo.config.pixel_end_token_id).nonzero(as_tuple=True)[0].item()

                eol_mask = (input_id ==self.wallaroo.config.eol_id)
                eol_mask[:pixel_start_index+1] = False
                eol_mask[pixel_end_index:] = False

                pixel_n_image_tokens = pixel_end_index - pixel_start_index - 1 - eol_mask.sum().item()
                
                pixel_image_embed = pixel_vae_values[bs_idx]
                pixel_n_image_features = pixel_image_embed.shape[0]

                if pixel_n_image_tokens != pixel_n_image_features:
                    raise ValueError(
                        f"Pixel image features and image tokens do not match: tokens: {pixel_n_image_tokens}, features {pixel_n_image_features}"
                    )

                pixel_image_mask = torch.zeros_like(input_id, dtype=torch.bool)
                pixel_image_mask[pixel_start_index+1:pixel_end_index] = True
                pixel_image_mask[eol_mask] = False
                pixel_image_mask = pixel_image_mask.unsqueeze(-1).expand_as(input_embeds).to(input_embeds.device)
                
                pixel_image_embed = pixel_image_embed.to(t2i_input_embeds.device, t2i_input_embeds.dtype)
                input_embeds = input_embeds.masked_scatter(pixel_image_mask, pixel_image_embed)

            new_input_embeds.append(input_embeds)

        new_input_embeds = torch.stack(new_input_embeds)
        
        for i in range(0, img_token_count+end_of_line_num):

            if (i+1)%(token_per_row+1)!=0:

                outputs = self.wallaroo(inputs_embeds=new_input_embeds, attention_mask=t2i_attention_mask, position_ids=t2i_position_ids, output_hidden_states=True, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.hidden_states[-1]
                logits = self.edit_image_head(hidden_states)

                if cfg > 1.0:
                    logits, null_logits = torch.chunk(logits, 2, dim=0)
                    cfg_logits = null_logits[:, -1, :] + cfg * (logits[:, -1, :] - null_logits[:, -1, :])
                else:
                    cfg_logits = logits[:, -1, :]

                idx_next, _ = self.sample(cfg_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                result.append(idx_next)

                new_input_embeds = self.edit_input_adpater(vqvae_model.quantize.get_codebook_entry(idx_next).to(torch.bfloat16))
                t2i_attention_mask = torch.ones((B, 1)).to(i2i_inputs['input_ids'].device)
                t2i_position_ids = position_ids[:, :, i2i_inputs['input_ids'].shape[1]+i:i2i_inputs['input_ids'].shape[1]+i+1]
                
                if cfg > 1.0:
                    new_input_embeds = new_input_embeds.repeat(2, 1, 1)
                    t2i_attention_mask = t2i_attention_mask.repeat(2, 1)
                    null_t2i_position_ids = null_position_ids[:, :, i2i_inputs['input_ids'].shape[1]+i:i2i_inputs['input_ids'].shape[1]+i+1]  ## different, can not repeat
                    t2i_position_ids = torch.cat([t2i_position_ids, null_t2i_position_ids], dim=1)

            # add line
            else:
                ## input and save as kv cache
                outputs = self.wallaroo(inputs_embeds=new_input_embeds, attention_mask=t2i_attention_mask, position_ids=t2i_position_ids, output_hidden_states=True, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)

                new_input_embeds = self.wallaroo.model.embed_tokens(torch.ones((B, 1)).to(i2i_inputs['input_ids'].device).fill_(eol_id).long())
                t2i_attention_mask = torch.ones((B, 1)).to(i2i_inputs['input_ids'].device)
                t2i_position_ids = position_ids[:, :, i2i_inputs['input_ids'].shape[1]+i:i2i_inputs['input_ids'].shape[1]+i+1]
                
                if cfg > 1.0:
                    new_input_embeds = new_input_embeds.repeat(2, 1, 1)
                    t2i_attention_mask = t2i_attention_mask.repeat(2, 1)
                    null_t2i_position_ids = null_position_ids[:, :, i2i_inputs['input_ids'].shape[1]+i:i2i_inputs['input_ids'].shape[1]+i+1]  ## different, can not repeat
                    t2i_position_ids = torch.cat([t2i_position_ids, null_t2i_position_ids], dim=1)

        return torch.cat(result, dim=1)


    

    # @torch.no_grad() #### Note that this is a very standard image editing process at cost of time consumption
    # def mar_i2i_generate(self, t2i_inputs, null_t2i_inputs, vqvae_model, eot_token, cfg=1.0, temperature=1.0, top_k=2000, top_p=1.0, generate_start_id=None, generate_end_id=None, eol_id=151643, tgt_h = 384, tgt_w=384, ds=16):
        
    #     result = [] 
    #     continue_features = []

    #     generate_img_pad_id = self.wallaroo.config.generate_img_pad_id
    #     B, device = t2i_inputs['input_ids'].shape[0], t2i_inputs['input_ids'].device

    #     img_token_count = (tgt_w//ds) * (tgt_h//ds)
    #     token_per_row = tgt_w//ds

    #     if cfg > 1.0:
    #         t2i_input_ids = torch.cat([t2i_inputs['input_ids'], null_t2i_inputs['input_ids']], dim=0)
    #         t2i_attention_mask = torch.cat([t2i_inputs['attention_mask'], null_t2i_inputs['attention_mask']], dim=0)
    #         image_grid_thw = [t2i_inputs['image_grid_thw'], null_t2i_inputs['image_grid_thw']]
    #         pixel_values = [t2i_inputs['pixel_values'][0], null_t2i_inputs['pixel_values'][0]]
    #         pixel_vae_values = [t2i_inputs['pixel_vae_values'][0], null_t2i_inputs['pixel_vae_values'][0]]
    #     else:
    #         t2i_input_ids = t2i_inputs['input_ids']
    #         t2i_attention_mask = t2i_inputs['attention_mask']
    #         image_grid_thw = [t2i_inputs['image_grid_thw']]
    #         pixel_values = t2i_inputs['pixel_values']
    #         pixel_vae_values = t2i_inputs['pixel_vae_values']

    #     for i in range(1, img_token_count+1):

    #         outputs = self.wallaroo(input_ids=t2i_input_ids, 
    #                                 attention_mask=t2i_attention_mask, 
    #                                 image_grid_thw=image_grid_thw, 
    #                                 pixel_values=pixel_values, 
    #                                 pixel_vae_values=pixel_vae_values, 
    #                                 output_hidden_states=True, 
    #                                 continue_vq_features=torch.cat(continue_features, dim=1) if len(continue_features)!=0 else None)

    #         hidden_states = outputs.hidden_states[-1]
    #         logits = self.gen_image_head(hidden_states)
            
    #         if cfg > 1.0:
    #             logits, null_logits = torch.chunk(logits, 2, dim=0)
    #             cfg_logits = null_logits[:, -1, :] + cfg * (logits[:, -1, :] - null_logits[:, -1, :])
    #         else:
    #             cfg_logits = logits[:, -1, :]

    #         idx_next, _ = self.sample(cfg_logits, temperature=temperature, top_k=top_k, top_p=top_p)
    #         # print("predict idx:", idx_next)
    #         result.append(idx_next)
            
    #         if cfg > 1.0:
    #             t2i_input_ids = torch.cat([t2i_input_ids, torch.ones((B*2, 1)).to(t2i_inputs['input_ids'].device).fill_(generate_img_pad_id).long()], dim=1)
    #             t2i_attention_mask = torch.cat([t2i_attention_mask, torch.ones((B*2, 1)).to(t2i_inputs['input_ids'].device)], dim=1)
    #             continue_features.append(self.gen_input_adpater(vqvae_model.quantize.get_codebook_entry(idx_next).to(torch.bfloat16)).repeat(B*2, 1, 1))
    #         else:
    #             t2i_input_ids = torch.cat([t2i_input_ids, torch.ones((B, 1)).to(t2i_inputs['input_ids'].device).fill_(generate_img_pad_id).long()], dim=1)
    #             t2i_attention_mask = torch.cat([t2i_attention_mask, torch.ones((B, 1)).to(t2i_inputs['input_ids'].device)], dim=1)
    #             continue_features.append(self.gen_input_adpater(vqvae_model.quantize.get_codebook_entry(idx_next).to(torch.bfloat16)))

    #         ## add eol token
    #         if i%token_per_row==0 and i!=img_token_count:
    #             if cfg > 1.0:
    #                 t2i_input_ids = torch.cat([t2i_input_ids, torch.ones((B*2, 1)).to(t2i_inputs['input_ids'].device).fill_(eol_id).long()], dim=1)
    #                 t2i_attention_mask = torch.cat([t2i_attention_mask, torch.ones((B*2, 1)).to(t2i_inputs['input_ids'].device)], dim=1)
    #             else:
    #                 t2i_input_ids = torch.cat([t2i_input_ids, torch.ones((B, 1)).to(t2i_inputs['input_ids'].device).fill_(eol_id).long()], dim=1)
    #                 t2i_attention_mask = torch.cat([t2i_attention_mask, torch.ones((B, 1)).to(t2i_inputs['input_ids'].device)], dim=1)

    #     return torch.cat(result, dim=1)

