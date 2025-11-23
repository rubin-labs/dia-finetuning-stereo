# Dia TTS Model Fine-Tuning

A training pipeline for fine-tuning the **Dia** TTS model using Hugging Face datasets or local audio–text pairs. Supports mixed-precision, model compilation, 8-bit optimizers, streaming datasets, and evaluation via TensorBoard.
For multilingual training, the pipeline supports language-tags ```[iso_code]```.For training a multilingual model, you have to provide a dataset with a language column containing the iso_code.


---


## Installation

```bash
git clone https://github.com/stlohrey/dia-finetuning.git
cd dia-finetuning
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Usage Example

```bash
python -m dia.finetune \
  --config path/to/dia/config.json \
  --dataset Paradoxia/opendata-iisys-hui \
  --hub_model nari-labs/Dia-1.6B \
  --run_name my_experiment \
  --output_dir ./checkpoints \
```

---

## Configuration

* **JSON Config**: `dia/config.json` defines model sizes, token lengths, delay patterns, and audio PAD/BOS/EOS values.
* **TrainConfig**: Default hyperparameters (epochs, batch size, learning rate, warmup, logging & saving steps, etc.) are set in the finetuning script in `TrainConfig`.
* **CLI Config**: train settings can be passed via `train.py` flags (see below).

---

## Major CLI Arguments

| Argument                | Type   | Default                        | Description                                                      |                                    |
| ----------------------- | ------ | ------------------------------ | ---------------------------------------------------------------- | ---------------------------------- |
| `--config`              | `Path` | `dia/config.json`              | Path to the Dia JSON config.                                     |                                    |
| `--dataset`             | `str`  | `Paradoxia/opendata-iisys-hui` | HF dataset name (train split).                                   |                                    |
| `--dataset2`            | `str`  | `None`                         | (Optional) Second HF dataset to interleave.                      |                                    |
| `--streaming`           | `bool` | `True`                         | Use HF streaming API.                                            |                                    |
| `--hub_model`           | `str`  | `nari-labs/Dia-1.6B`           | HF Hub repo for base checkpoint.                                 |                                    |
| `--local_ckpt`          | `str`  | `None`                         | Path to local model checkpoint (`.pth`).                         |                                    |
| `--csv_path`            | `Path` | `None`                         | CSV file with \`audio                                            | example.wav\|transcript format.    |
| `--audio_root`          | `Path` | `None`                         | Base directory for local audio files (required if `--csv_path`). |                                    |
| `--run_name`            | `str`  |                                | TensorBoard run directory name.                                  |                                    |
| `--output_dir`          | `Path` |                                | Directory for saving checkpoints.                                |                                    |
| `--shuffle_buffer_size` | `int`  | `None`                         | Buffer size for streaming shuffle.                               |                                    |
| `--seed`                | `int`  | `42`                           | Random seed for reproducibility.                                 |                                    |
| `--half`                | `bool` | `False`                        | Load model in FP16.                                              |                                    |
| `--compile`             | `bool` | `False`                        | Enable `torch.compile` (Inductor backend).                       |                                    |

---

## Using accelerate

Accelerate from HF is awesome it is like using all GPU without much settings.
First install the project and use *accelerate config*

Select multi-gpy , fp16, all gpu .. 

```
accelerate launch -m dia.finetune
--hub_model nari-labs/Dia-1.6B
--run_name your_name
--output_dir ..output_dir
--csv_path ..some//metadata_en.csv
--audio_root ..audio_files_path/wavs
```

Also you can set all args will remian same if, In above command i have not used datasets. \
Rememebr use all default args and dont use --half, also i notice on some systems if we pass config in arg we get error. so avoid as its already in fie.

## Monitoring & Evaluation

* **TensorBoard**:

  ```bash
  tensorboard --logdir runs
  ```

  * `Loss/train`, `Loss/eval`, learning rate, grad‐norm.
  * Audio samples for each test sentence in multiple languages, can be specified inside finetune.py.

* **Checkpoints**: Saved in `output_dir` as `ckpt_step{N}.pth` and `ckpt_epoch{E}.pth`.

---

## Inference (Gradio UI)

**Convert Checkpoint to fp32**

If you used --half and --compile during training, you have to unwrap and convert the checkpoint back to fp32:
```bash
./python -m dia.convert_ckpt \
  --input-ckpt /path/to/ckpt_epoch1.pth \
  --output-ckpt /path/to/ckpt_epoch1_fp32.pth \
  --config    /path/to/config.json
```

A Gradio-based web app for interactive text-to-speech synthesis. It provides sliders for generation parameters and accepts optional audio prompts.

 ```bash
python app_local.py \
  --local_ckpt path/to/ckpt_epoch1_fp32.pth \
  --config path/to/inference/config.json
```

Open the displayed URL in your browser to interact with the model.

---






<p align="center">
<a href="https://github.com/nari-labs/dia">
<img src="./dia/static/images/banner.png">
</a>
</p>
<p align="center">
<a href="https://tally.so/r/meokbo" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Join-Waitlist-white?style=for-the-badge"></a>
<a href="https://discord.gg/pgdB5YRe" target="_blank"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge"></a>
<a href="https://github.com/nari-labs/dia/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="LICENSE"></a>
</p>
<p align="center">
<a href="https://huggingface.co/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Dataset on HuggingFace" height=42 ></a>
<a href="https://huggingface.co/spaces/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>

Dia is a 1.6B parameter text to speech model created by Nari Labs.

Dia **directly generates highly realistic dialogue from a transcript**. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

To accelerate research, we are providing access to pretrained model checkpoints and inference code. The model weights are hosted on [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B). The model only supports English generation at the moment.

We also provide a [demo page](https://yummy-fir-7a4.notion.site/dia) comparing our model to [ElevenLabs Studio](https://elevenlabs.io/studio) and [Sesame CSM-1B](https://github.com/SesameAILabs/csm).

- (Update) We have a ZeroGPU Space running! Try it now [here](https://huggingface.co/spaces/nari-labs/Dia-1.6B). Thanks to the HF team for the support :)
- Join our [discord server](https://discord.gg/pgdB5YRe) for community support and access to new features.
- Play with a larger version of Dia: generate fun conversations, remix content, and share with friends. 🔮 Join the [waitlist](https://tally.so/r/meokbo) for early access.

## ⚡️ Quickstart

### Install via pip

```bash
# Install directly from GitHub
pip install git+https://github.com/nari-labs/dia.git
```

### Run the Gradio UI

This will open a Gradio UI that you can work on.

```bash
git clone https://github.com/nari-labs/dia.git
cd dia && uv run app.py
```

or if you do not have `uv` pre-installed:

```bash
git clone https://github.com/nari-labs/dia.git
cd dia
python -m venv .venv
source .venv/bin/activate
pip install -e .
python app.py
```

Note that the model was not fine-tuned on a specific voice. Hence, you will get different voices every time you run the model.
You can keep speaker consistency by either adding an audio prompt (a guide coming VERY soon - try it with the second example on Gradio for now), or fixing the seed.

## Features

- Generate dialogue via `[S1]` and `[S2]` tag
- Generate non-verbal like `(laughs)`, `(coughs)`, etc.
  - Below verbal tags will be recognized, but might result in unexpected output.
  - `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`
- Voice cloning. See [`example/voice_clone.py`](example/voice_clone.py) for more information.
  - In the Hugging Face space, you can upload the audio you want to clone and place its transcript before your script. Make sure the transcript follows the required format. The model will then output only the content of your script.

## ⚙️ Usage

### As a Python Library

```python
import soundfile as sf

from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

output = model.generate(text)

sf.write("simple.mp3", output, 44100)
```

A pypi package and a working CLI tool will be available soon.

## 💻 Hardware and Inference Speed

Dia has been tested on only GPUs (pytorch 2.0+, CUDA 12.6). CPU support is to be added soon.
The initial run will take longer as the Descript Audio Codec also needs to be downloaded.

On enterprise GPUs, Dia can generate audio in real-time. On older GPUs, inference time will be slower.
For reference, on a A4000 GPU, Dia roughly generates 40 tokens/s (86 tokens equals 1 second of audio).
`torch.compile` will increase speeds for supported GPUs.

The full version of Dia requires around 12-13GB of VRAM to run. We will be adding a quantized version in the future.

If you don't have hardware available or if you want to play with bigger versions of our models, join the waitlist [here](https://tally.so/r/meokbo).

## 🪪 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## 🔭 TODO / Future Work

- Docker support.
- Optimize inference speed.
- Add quantization for memory efficiency.

## 🤝 Contributing

We are a tiny team of 1 full-time and 1 part-time research-engineers. We are extra-welcome to any contributions!
Join our [Discord Server](https://discord.gg/pgdB5YRe) for discussions.

## 🤗 Acknowledgements

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- Hugging Face for providing the ZeroGPU Grant.
- "Nari" is a pure Korean word for lily.
- We thank Jason Y. for providing help with data filtering.


## ⭐ Star History

<a href="https://www.star-history.com/#nari-labs/dia&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
 </picture>
</a>
