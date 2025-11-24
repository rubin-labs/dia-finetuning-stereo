#!/usr/bin/env python3

import argparse
import json
import os

import torch
import torchaudio
from gptqmodel import GPTQModel
from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
from gptqmodel.models.base import BaseGPTQModel
from huggingface_hub import snapshot_download
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor

from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

QWEN_SAMPLE_RATE = 16000

QWEN_SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# Music analysis with vocal & drum detection
PROMPT = r"""Analyze this audio track:

Tasks:
1. Detect whether ANY vocals (singing, rapping, spoken word, humming, choir, etc.) are present.
2. Detect whether drums/percussion are present. Choose `"drums"` if you hear rhythmic percussion (acoustic or electronic), otherwise `"no drums"`.
3. Describe the musical style.

Output rules:
- Always respond with valid JSON.
- Always include a `"drums"` field with either `["drums"]` or `["no drums"]`.
- If vocals are detected, set `"vocals": ["vocals"]` and provide rich descriptive lists for `genre`, `subgenre`, `tempo`, and `mood`.
- If the track is purely instrumental, provide EXACTLY 2 `genre` tags (use the most defining ones) and EXACTLY 2 `mood` tags. `subgenre` and `tempo` are optional but helpful when confident.

Example (vocals detected):
```json
{
  "vocals": ["vocals"],
  "drums": ["drums"],
  "genre": ["pop", "dance-pop"],
  "subgenre": ["electropop", "modern pop production"],
  "tempo": ["energetic"],
  "mood": ["uplifting", "confident"]
}
```

Example (instrumental):
```json
{
  "drums": ["no drums"],
  "genre": ["ambient", "soundscape"],
  "mood": ["soothing", "ethereal"]
}
```"""


@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)
    model = cls._from_config(config, **kwargs)
    return model


Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config


class Qwen2_5OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = [
        "thinker.model.embed_tokens",
        "thinker.model.norm",
        "thinker.audio_tower",
        "thinker.model.rotary_emb",
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    def pre_quantize_generate_hook_start(self):
        self.thinker.audio_tower = self.thinker.audio_tower.to(
            self.quantize_config.device
        )

    def pre_quantize_generate_hook_end(self):
        self.thinker.audio_tower = self.thinker.audio_tower.to("cuda")

    def preprocess_dataset(self, sample):
        return sample


MODEL_MAP["qwen2_5_omni"] = Qwen2_5OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])


def load_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)

    device_map = {
        "thinker.model": "cuda",
        "thinker.lm_head": "cuda",
        # "thinker.visual": "cpu", 
        "thinker.audio_tower": "cuda",
        "talker": "cpu",
        "token2wav": "cpu",
    }

    model = GPTQModel.load(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.disable_talker()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def read_audio(file_path):
    audio, sr = torchaudio.load(file_path)
    audio = audio[:, : sr * 360]
    if sr != QWEN_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, QWEN_SAMPLE_RATE)
        sr = QWEN_SAMPLE_RATE
    audio = audio.mean(dim=0, keepdim=True)
    return audio, sr


def inference(file_path, model, processor):
    audio, _ = read_audio(file_path)
    audio = audio.numpy().squeeze(axis=0)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    # Copy tensors to GPU and match dtypes
    ks = list(inputs.keys())
    for k in ks:
        if hasattr(inputs[k], "to"):
            inputs[k] = inputs[k].to("cuda")
            if inputs[k].dtype.is_floating_point:
                inputs[k] = inputs[k].to(model.dtype)

    output_ids = model.thinker.generate(
        **inputs,
        max_new_tokens=1000,
        use_audio_in_video=False,
    )

    generate_ids = output_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


def parse_prompt(content):
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()
    
    # Try to fix truncated JSON
    if not content.endswith('}'):
        # Find the last complete field
        content = content.rstrip('", \n\t')
        if content.count('"') % 2 == 1:  # Odd quotes means incomplete
            content += '"'
        content += '}'

    prompt = ""
    try:
        tags = []
        data = json.loads(content)
        
        # Check if vocals are detected
        if "vocals" in data:
            # For vocal tracks, just return "voice"
            return "voice"
        
        # For instrumental tracks, combine all tags
        ordered_sources = [
            data.get("genre", []),
            data.get("subgenre", []),
            data.get("mood", []),
            data.get("tempo", []),
        ]
        for source in ordered_sources:
            tags += source

        tags = [x.strip().lower() for x in tags if x.strip()]

        # Preserve insertion order while removing duplicates
        unique_tags = []
        seen = set()
        for tag in tags:
            if tag not in seen:
                unique_tags.append(tag)
                seen.add(tag)

        # Limit to six descriptive tags to keep prompts concise
        prompt_tags = unique_tags[:6]
        prompt = ", ".join(prompt_tags)

    except Exception:
        print("Failed to parse content")
        print(content)

    return prompt


def do_files(data_dir, overwrite):
    model, processor = load_model("Qwen/Qwen2.5-Omni-7B-GPTQ-Int4")

    # Create prompts directory parallel to audio directory
    # Handle trailing slash issue
    data_dir_clean = data_dir.rstrip('/')
    parent_dir = os.path.dirname(data_dir_clean)
    prompts_dir = os.path.join(parent_dir, "audio_prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    
    print(f"Saving prompts to: {prompts_dir}")

    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".wav",
    }

    for file in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(file)
        if ext.lower() not in extensions:
            continue

        file_path = os.path.join(data_dir, file)
        prompt_path = os.path.join(prompts_dir, stem + "_prompt.txt")

        need_prompt = overwrite or (not os.path.exists(prompt_path))

        if not need_prompt:
            continue

        print(file)
        content = inference(file_path, model, processor)
        prompt = parse_prompt(content)

        with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(prompt)


@torch.inference_mode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="jamendomaxcaps_sample")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    do_files(
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
