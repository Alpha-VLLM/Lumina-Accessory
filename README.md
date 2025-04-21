<div align="center">
<h1> Lumina-Accessory: Instruction Fine-tuned Rectified Flow Transformer for Universial Image Generation </h1>

[![Lumina-Next](https://img.shields.io/badge/Paper-Lumina--Image--2.0-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2503.21758)&#160;
[![Badge](https://img.shields.io/badge/-WeChat@Join%20Our%20Group-000000?logo=wechat&logoColor=07C160)](https://github.com/ChinChyi/ipictures/blob/main/20250421.jpg?raw=true)&#160;
[![Static Badge](https://img.shields.io/badge/Lumina--Accessory%20checkpoints-Model(2B)-yellow?logoColor=violet&label=%F0%9F%A4%97%20Lumina-Accessory%20checkpoints)](https://huggingface.co/Alpha-VLLM/Lumina-Accessory)

<p align="center">
 <img src="./assets/Illustration.png" width="100%"/>
 <br>
</p>

</div>
<div align="center">

</div>

## ✨ Features

**Lumina-Accessory** is a multi-task instruction fine-tuning framework designed for Lumina series (currently supporting **Lumina-Image-2.0**). This repository includes:  

- **🧠 Tuning Code** – Unifies various image-to-image tasks in a **sequence concatenation manner**, supporting both **universal** and **task-specific** model tuning.  

- **⚖️ Instruction Fine-tuned Universal Model Weights** – Initialized from **Lumina-Image-2.0**, supporting:  
  - 🖼️ **Spatial conditional generation**  
  - 🔧 **Infilling & Restoration**  
  - 💡 **Relighting**  
  - 🎨 **Subject-driven generation**  
  - ✏️ **Instruction-based editing**  

- **🚀 Inference Code & Gradio Demo** – Test and showcase the **universal model’s capabilities** interactively!  

## 📰 News
- [2025-4-21] 🚀🚀🚀 We are excited to release `Lumina-Accessory`, including:
  - 🎯 Checkpoints, Fine-Tuning and Inference code.

## 📑 Open-source Plan

 - [x] Tuning code
 - [x] Inference Code
 - [x] Checkpoints
 - [ ] Web Demo (Gradio)

## 🏠 Architecture
✨ Lumina-Accessory directly leverages the self-attention mechanism in DiT to perform interaction between condition and target image tokens, consistent with approaches such as [OminiControl](https://github.com/Yuanshi9815/OminiControl), [DSD](https://primecai.github.io/dsd/), [VisualCloze](https://github.com/lzyhha/VisualCloze), etc.

✨ Built on top of Lumina-Image-2.0, Lumina-Accessory introduces an additional condition processor, initialized with the weights of the latent processor.

✨ Similar to [OminiControl](https://github.com/Yuanshi9815/OminiControl), we modulate both condition and target image tokens with different time conditions, and apply distinct positional embeddings for different types of conditions.

<p align="center">
 <img src="./assets/Architecture.png" width="100%"/>
 <br>
</p>

## 🎮 Model Zoo

| Resolution | Parameter| Text Encoder | VAE | Download URL  |
| ---------- | ----------------------- | ------------ | -----------|-------------- |
| 1024       | 2.6B             |    [Gemma-2-2B](https://huggingface.co/google/gemma-2-2b)  |   [FLUX-VAE-16CH](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae) | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-Accessory) |

## 📊 Model Capability

| Task Type                  | Training Data               | Model Ability               |
|----------------------------|-----------------------------|-----------------------------|
| **Spatial Conditional Generation** | Internal Data    | 😄 (Good)           |
| **Infilling & Restoration** | Internal Data     | 😄 (Good)          |
| **Relighting**             | IC-Light Synthetic Data | 😊 (Moderate)            |
| **Subject-Driven Generation** | Subject200K | 😐 (Basic)                 |
| **Instruction-Based Editing** | OmniEdit-1.2M    | 😐 (Basic)                 |

## 💻 Finetuning Code
### 1. Create a conda environment and install PyTorch
```bash
conda create -n Lumina2 -y
conda activate Lumina2
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
### 2.Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Install flash-attn
```bash
pip install flash-attn --no-build-isolation
```
### 4. Prepare data
You can place the links to your data files in `./configs/data.yaml`. 

For tasks where the condition can be generated online, your image-text pair training data format should adhere to the following:
```json
{
    "image_path": "path/to/your/image",
    "prompt": "a description of the image"
}
```

For tasks that require loading a condition image, the training data format should be as follows:
```json
{
    "input_image": "path/to/your/condition",
    "output_image": "path/to/your/target",
    "prompt": "a description of the image"
}
```
### 5. Start finetuning
```bash
bash scripts/run_1024_finetune.sh
```
## 🚀 Inference Code
We support multiple solvers including Midpoint Solver, Euler Solver, and **DPM Solver** for inference.
> [!Note]
> Both the Gradio demo and the direct inference method use the .pth format weight file, which can be downloaded from [huggingface](https://huggingface.co/Alpha-VLLM/Lumina-Accessory). We have uploaded the .pth weight files, and you can simply specify the `--ckpt` argument as the download directory.

> [!Note]
> The code has just been cleaned up, if there are any issues please let us know.

- Direct Inference
```python   
NUM_STEPS=50
CFG_SCALE=4.0
TIME_SHIFTING_FACTOR=6
SEED=20
SOLVER=euler
TASK_TYPE="Image Infilling"
CAP_DIR=./examples/caption_list.json
OUT_DIR=./examples/outputs
MODEL_CHECKPOINT=/path/to/your/ckpt

python -u sample_accessory.py --ckpt ${MODEL_CHECKPOINT} \
--image_save_path ${OUT_DIR} \
--solver ${SOLVER} \
--num_sampling_steps ${STEPS} \
--caption_path ${CAP_DIR} \
--seed ${SEED} \
--time_shifting_factor ${TIME_SHIFTING_FACTOR} \
--cfg_scale ${CFG_SCALE} \
--batch_size 1 \
--rank 0 \
--task_type "${TASK_TYPE}"
```

- Gradio Demo
```python   
PRECISION="bf16" 
SOLVER="euler"
VAE="flux"
SHARE=False
MODEL_CHECKPOINT=/path/to/your/ckpt

torchrun --nproc_per_node=1 --master_port=18187 gradio_demo.py \
  --ckpt "$MODEL_CHECKPOINT" \
  --precision "$PRECISION" \
  --solver "$SOLVER" \
  --vae "$VAE" \
  --share "$SHARE"
```

<p align="left">
 <img src="./assets/Demo.png" width="70%"/>
 <br>
</p>

## Citation

If you find the provided code or models useful for your research, consider citing them as:

```bib
@Misc{lumina-accessory,
  author = {Alpha-VLLM Team},
  title  = {Lumina-Accessory GitHub Page},
  year   = {2025},
}
```

## Related Work
[Lumina-Image 2.0: A Unified and Efficient Image Generative Framework](https://github.com/Alpha-VLLM/Lumina-Image-2.0)

[OminiControl: Minimal and Universal Control for Diffusion Transformer](https://github.com/Yuanshi9815/OminiControl)

[Diffusion Self-Distillation for Zero-Shot Customized Image Generation](https://primecai.github.io/dsd/)

[OmniEdit: Building Image Editing Generalist Models Through Specialist Supervision](https://tiger-ai-lab.github.io/OmniEdit/)

[VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](https://github.com/lzyhha/VisualCloze)

[OminiControl2: Efficient Conditioning for Diffusion Transformers](https://arxiv.org/abs/2503.08280)
