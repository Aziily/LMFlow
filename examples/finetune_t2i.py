import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"
import shutil
from pathlib import Path
import gc

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    SD3Transformer2DModel
)
from transformers import (
    PretrainedConfig,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5TokenizerFast
)
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import HfArgumentParser
from peft import LoraConfig

from lmflow.args import (
    DiffuserModelArguments, 
    T2IDatasetArguments, 
    AutoArguments,
)
from lmflow.datasets import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline

def import_model_class_from_model_name_or_path(
    model_name_or_path: str, 
    subfolder: str = "text_encoder",
    revision = None,
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def main():
    pipeline_name = "diffuser_tuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
    
    parser = HfArgumentParser((DiffuserModelArguments, T2IDatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
    
    logging_dir = Path(pipeline_args.output_dir, pipeline_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=pipeline_args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=pipeline_args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process and pipeline_args.overwrite_output_dir and os.path.exists(pipeline_args.output_dir):
        shutil.rmtree(pipeline_args.output_dir)
    
    """
    Preprocess dataset
    """
    if data_args.preprocessor_kind == "simple":
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_args.model_name_or_path, subfolder="text_encoder").to(accelerator.device)
        vae = AutoencoderKL.from_pretrained(model_args.model_name_or_path, subfolder="vae").to(accelerator.device)
        
        # dataset = build_t2i_dataset(data_args, tokenizer, text_encoder, vae)
        kwargs = {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae}
        dataset = Dataset(data_args, backend="t2i", **kwargs)
        
        del tokenizer, text_encoder, vae
    elif data_args.preprocessor_kind == "SD3":
        vae = AutoencoderKL.from_pretrained(
            model_args.model_name_or_path,
            subfolder="vae",
        )
        
        tokenizer_one = CLIPTokenizer.from_pretrained(
            model_args.model_name_or_path,
            subfolder="tokenizer",
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            model_args.model_name_or_path,
            subfolder="tokenizer_2",
        )
        tokenizer_three = T5TokenizerFast.from_pretrained(
            model_args.model_name_or_path,
            subfolder="tokenizer_3",
        )
        def load_encoders():
            text_encoder_one = import_model_class_from_model_name_or_path(
                model_args.model_name_or_path, subfolder="text_encoder"
            ).from_pretrained(
                model_args.model_name_or_path, subfolder="text_encoder"
            ).to(accelerator.device)
            text_encoder_two = import_model_class_from_model_name_or_path(
                model_args.model_name_or_path, subfolder="text_encoder_2"
            ).from_pretrained(
                model_args.model_name_or_path, subfolder="text_encoder_2"
            ).to(accelerator.device)
            text_encoder_three = import_model_class_from_model_name_or_path(
                model_args.model_name_or_path, subfolder="text_encoder_3"
            ).from_pretrained(
                model_args.model_name_or_path, subfolder="text_encoder_3"
            ).to(accelerator.device)
            return text_encoder_one, text_encoder_two, text_encoder_three
        text_encoder_one, text_encoder_two, text_encoder_three = load_encoders()
        
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
        
        kwargs = {"tokenizers": tokenizers, "text_encoders": text_encoders, "vae": vae}
        dataset = Dataset(data_args, backend="t2i", **kwargs)
        
        del tokenizer_one, tokenizer_two, tokenizer_three, text_encoder_one, text_encoder_two, text_encoder_three, vae
    
    torch.cuda.empty_cache()
    gc.collect()
    
    model = None
    if model_args.arch_type == "unet":
        model = UNet2DConditionModel.from_pretrained(model_args.model_name_or_path, subfolder=model_args.arch_type)
    elif model_args.arch_type == "transformer":
        raise NotImplementedError("Transformer model is not implemented.")
    elif model_args.arch_type == "SD3transformer2D":
        model = SD3Transformer2DModel.from_pretrained(model_args.model_name_or_path, subfolder="transformer")
        pipeline_args.do_valid = False
        pipeline_args.do_test = False
    else:
        raise ValueError("The model type is not supported.")
    
    if model_args.use_lora:
        accelerator.print(f"Using LoRA of {model_args.lora_target_modules} for training")
        model.requires_grad_(False)
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=model_args.lora_target_modules,
        )
        model.add_adapter(lora_config)
    else:
        model.requires_grad_(True)
    
    fintuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    accelerator.init_trackers("text2image-finetune", config={
        "data_args": data_args,
        "model_args": model_args,
        "pipeline_args": pipeline_args,
    })
    
    accelerator.wait_for_everyone()
    fintuner.tune(
        accelerator=accelerator,
        model=model, dataset=dataset
    )

if __name__ == '__main__':
    main()
