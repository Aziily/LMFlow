#!/usr/bin/env python
# coding=utf-8

"""This Python code defines a class T2I Dataset.
"""
import json
from PIL import Image
import os.path as osp
from tqdm import tqdm
import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from lmflow.args import T2IDatasetArguments

logger = logging.getLogger(__name__)

class CustomT2IDataset(Dataset):
    """
    Dataset for T2I data
    
    Parameters
    ------------
    data_args: T2IDatasetArguments
        The arguments for the dataset.
    """
    
    def __init__(self, data_args: T2IDatasetArguments):
        self.data_args = data_args
        self.image_folder = osp.join(data_args.dataset_path, data_args.image_folder)
        self.data_file = osp.join(data_args.dataset_path, data_args.train_file)
        
        self.data_dict = json.load(open(self.data_file, "r"))
        assert self.data_dict["type"] == "image_text", "The dataset type must be text-image."
        
        self.data_instances = self.data_dict["instances"]
    
    def __len__(self):
        return len(self.data_instances)
    
    def __getitem__(self, idx):
        instance = self.data_instances[idx]
        image_path = osp.join(self.image_folder, instance["images"])
        image = Image.open(image_path)
        image = image.convert("RGB")
        
        return {
            "image": image,
            "text": instance["text"],
        }

class EncodePreprocessor(object):
    """
    This class implement the preparation of the data for the model.
    For different Diffusion model, the preparation is different.
    
    Parameters
    ------------
    data_args: T2IDatasetArguments
        The arguments for the dataset.
    
    **kwargs
        The arguments for the preprocessor.
        
    Example
    ------------
    >>> data_args.preprocessor_kind
    simple
    >>> kwargs = {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae}
    >>> raw_dataset = CustomT2IDataset(data_args)
    >>> preprocessor = EncodePreprocessor(data_args=data_args, **kwargs)
    >>> dataset = PreprocessedT2IDataset(raw_dataset, data_args, preprocessor)
    """
    
    def __init__(self, data_args: T2IDatasetArguments, 
                 **kwargs):
        self.transform = transforms.Compose(
            [
                transforms.Resize(data_args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(data_args.image_size) if data_args.image_crop_type == "center" else transforms.RandomCrop(data_args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        
        self.pre_func = None
        if data_args.preprocessor_kind == "simple":
            self.register_simple_func(**kwargs)
        elif data_args.preprocessor_kind == "SD3":
            self.register_sd3_func(**kwargs)
        else:
            raise NotImplementedError(f"The preprocessor kind {data_args.preprocessor_kind} is not implemented.")
    
    def register_simple_func(self, 
                             tokenizer, 
                             text_encoder, 
                             vae):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        
        def simple_func(data_item):
            image = self.transform(data_item["image"])
            latents = self.vae.encode(image.to(self.vae.device, dtype=self.vae.dtype).unsqueeze(0)).latent_dist.sample()
            encoded_image = latents * self.vae.config.scaling_factor
            encoded_image = encoded_image.detach()
            encoded_image=encoded_image.squeeze(0).cpu()
            
            max_length = self.tokenizer.model_max_length
            tokens = self.tokenizer([data_item["text"]], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
            encoded_text = self.text_encoder(tokens.to(self.text_encoder.device))[0]
            encoded_text = encoded_text.detach()
            encoded_text =encoded_text.squeeze(0).cpu()
            
            return {
                "image": encoded_image,
                "text": encoded_text,
            }
            
        self.pre_func = simple_func
        
    def register_sd3_func(self,
                          tokenizers,
                          text_encoders,
                          vae):
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders
        self.vae = vae
        
        @torch.no_grad()
        def encode_prompt(
            text_encoders,
            tokenizers,
            prompt: str,
            max_sequence_length,
            device=None,
            num_images_per_prompt: int = 1,
        ):
            
            def _encode_prompt_with_t5(
                text_encoder,
                tokenizer,
                max_sequence_length,
                prompt=None,
                num_images_per_prompt=1,
                device=None,
            ):
                prompt = [prompt] if isinstance(prompt, str) else prompt
                batch_size = len(prompt)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(text_input_ids.to(device))[0]

                dtype = text_encoder.dtype
                prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

                _, seq_len, _ = prompt_embeds.shape

                # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

                return prompt_embeds

            def _encode_prompt_with_clip(
                text_encoder,
                tokenizer,
                prompt: str,
                device=None,
                num_images_per_prompt: int = 1,
            ):
                prompt = [prompt] if isinstance(prompt, str) else prompt
                batch_size = len(prompt)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                _, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

                return prompt_embeds, pooled_prompt_embeds
            
            prompt = [prompt] if isinstance(prompt, str) else prompt

            clip_tokenizers = tokenizers[:2]
            clip_text_encoders = text_encoders[:2]

            clip_prompt_embeds_list = []
            clip_pooled_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
                prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device if device is not None else text_encoder.device,
                    num_images_per_prompt=num_images_per_prompt,
                )
                clip_prompt_embeds_list.append(prompt_embeds)
                clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

            clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

            t5_prompt_embed = _encode_prompt_with_t5(
                text_encoders[-1],
                tokenizers[-1],
                max_sequence_length,
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device if device is not None else text_encoders[-1].device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

            return prompt_embeds, pooled_prompt_embeds
        
        def sd3_func(data_item):
            image = self.transform(data_item["image"])
            latents = self.vae.encode(image.to(self.vae.device, dtype=self.vae.dtype).unsqueeze(0)).latent_dist.sample()
            encoded_image = latents * self.vae.config.scaling_factor
            encoded_image = encoded_image.detach()
            encoded_image=encoded_image.squeeze(0).cpu()
            
            max_length = 77
            encoded_text, pooled_encoded_text = encode_prompt(
                self.text_encoders,
                self.tokenizers,
                data_item["text"],
                max_length,
                device=self.text_encoders[0].device,
                num_images_per_prompt=1,
            )
            encoded_text = encoded_text.detach().squeeze(0).cpu()
            pooled_encoded_text = pooled_encoded_text.detach().squeeze(0).cpu()
            
            return {
                "image": encoded_image,
                "text": encoded_text,
                "pool_text": pooled_encoded_text,
            }
            
        self.pre_func = sd3_func
        
    def __call__(self, data_item):
        return self.pre_func(data_item)     
   
class PreprocessedT2IDataset(Dataset):
    "Preprocess dataset with prompt"
    
    def __init__(self, raw_dataset:Dataset, 
                 data_args: T2IDatasetArguments, 
                 preprocessor:EncodePreprocessor):
        del data_args # Unused variable
        self.data_dict = []
        
        logger.info("Preprocessing data ...")
        for data_item in tqdm(raw_dataset):
            self.data_dict.append(preprocessor(data_item))
            
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        return self.data_dict[idx]

def build_t2i_dataset(data_args: T2IDatasetArguments, 
                      **kwargs):
    raw_dataset = CustomT2IDataset(data_args)
    preprocessor = EncodePreprocessor(data_args=data_args, **kwargs)
    dataset = PreprocessedT2IDataset(raw_dataset, data_args, preprocessor)
    
    return dataset