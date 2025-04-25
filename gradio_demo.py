import argparse
import json
import math
import os
import random
import socket
import time

from diffusers.models import AutoencoderKL
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import cv2
from transformers import pipeline

import torch.nn.functional as F

from data import DataNoReportException, ItemProcessor, MyDataset, read_general
from torchvision import transforms

import models_accessory as models
from transport import Sampler, create_transport
from models_accessory.lora import replace_linear_with_lora
import gradio as gr

#############################################################################
#                            Condition Generator                            #
#############################################################################
# Global variables to store loaded models
depth_pipeline = None
model = None
vae = None
text_encoder = None
tokenizer = None
sampler = None

# System prompt options
SYSTEM_PROMPTS = {
    "Text-to-Image": "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> ",
    "Image Infilling": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a partially masked image. <Prompt Start> ",
    "Image Restoration": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a degraded image. <Prompt Start> ",
    "Edge Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a canny edge condition. <Prompt Start> ",
    "Depth Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a depth map condition. <Prompt Start> ",
    "Brightness Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a brightness map condition. <Prompt Start> ",
    "Palette Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a palette map condition. <Prompt Start> ",
    "Human Keypoint Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a human keypoint condition. <Prompt Start> ",
    "Subject-driven": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and an object reference. <Prompt Start> ",
    "Image Relighting": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on image-lighting instructions and an original image. <Prompt Start> ",
    "Image Editing": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on editing instructions and an original image. <Prompt Start> ",
}

def get_canny_edges(image):
    canny_edge = cv2.Canny(np.array(image)[:, :, ::-1].copy(), 100, 200)
    canny_edge = torch.tensor(canny_edge)/255.0
    canny_edge = Image.fromarray((canny_edge.squeeze().numpy() * 255).astype(np.uint8))
    return canny_edge

def get_depth_map(image):
    global depth_pipeline
    if depth_pipeline is None:
        depth_pipeline = pipeline(
                            task="depth-estimation",
                            model="LiheYoung/depth-anything-small-hf",
                            device="cuda",
                            torch_dtype=torch.float32
        )
    source_image = image.convert("RGB")
    with torch.cuda.amp.autocast(enabled=False):
        depth_output = depth_pipeline(source_image)
    return depth_output["depth"].convert("RGB")

def get_brightness_map(image):
    image = image.filter(ImageFilter.GaussianBlur(6))
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lower_thresh = 85  
    upper_thresh = 170 
    brightness_map = np.zeros_like(gray_image)
    brightness_map[gray_image <= lower_thresh] = 0 
    brightness_map[(gray_image > lower_thresh) & (gray_image <= upper_thresh)] = 128 
    brightness_map[gray_image > upper_thresh] = 255 
    brightness_map = Image.fromarray(brightness_map.astype(np.uint8))
    return brightness_map

def get_palette_map(image, num_colors=8):
    image = image.filter(ImageFilter.GaussianBlur(12))
    w, h = image.size
    small_img = image.resize((w // 32, h // 32), Image.Resampling.NEAREST)
    
    img_array = np.array(small_img)
    pixels = img_array.reshape(-1, 3)
    
    pixels = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_array.shape)
    
    palette_map = Image.fromarray(quantized)
    palette_map = palette_map.resize((w, h), Image.Resampling.NEAREST) 
    
    return palette_map

#############################################################################
#                            Data item Processor                            #
#############################################################################

def resize_with_aspect_ratio(img, resolution=1024, divisible=64, aspect_ratio=None):
    """resize the image with aspect ratio, keep the area close to resolution**2, and the width and height can be divisible by divisible"""
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        if img.dim() == 3:
            c, h, w = img.shape
            batch_dim = False
        else:
            b, c, h, w = img.shape
            batch_dim = True
    else:
        w, h = img.size
        
    if aspect_ratio is None:
        aspect_ratio = w / h
    target_area = resolution * resolution
    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)
    
    new_w = max(new_w // divisible, 1) * divisible
    new_h = max(new_h // divisible, 1) * divisible
    
    if is_tensor:
        mode = 'bilinear'
        align_corners = False
        if batch_dim:
            return F.interpolate(img, size=(new_h, new_w), 
                               mode=mode, align_corners=align_corners)
        else:
            return F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                               mode=mode, align_corners=align_corners).squeeze(0)
    else:
        return img.resize((new_w, new_h), Image.LANCZOS)

class NonRGBError(DataNoReportException):
    pass

def encode_prompt(prompt_batch, text_encoder, tokenizer):
    """encode the text prompt"""
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt_batch,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

def load_models(args):
    print("Loading models...")
    global model, vae, text_encoder, tokenizer, sampler
    
    # Set data type
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.padding_side = "right"
    print("Tokenizer loaded")
    
    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2-2b", torch_dtype=dtype, device_map="cuda"
    ).eval()
    print("Text encoder loaded")
    cap_feat_dim = text_encoder.config.hidden_size
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device)
    vae.requires_grad_(False)
    print("VAE loaded")
    # Load training parameters
    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"), weights_only=False)
    
    # Create model
    model = models.__dict__[train_args.model](
        in_channels=16,
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().to(device, dtype=dtype)
    print("Model created")
    # Load model weights
    ckpt = torch.load(os.path.join(args.ckpt, f"consolidated{'_ema' if args.ema else ''}.00-of-01.pth"))
    model.load_state_dict(ckpt, strict=True)
    print("Model weights loaded")   
    # Create sampler
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    
    print("All models loaded successfully")

def process_image(image, resolution=1024):
    if image is None:
        return None
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = resize_with_aspect_ratio(image, resolution)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    return transform(image)

def generate_image(cond_image, prompt, system_prompt, num_steps=50, cfg_scale=4.0, t_shift=6, seed=None):
    """Main function for image generation"""
    global model, vae, text_encoder, tokenizer, sampler
    
    if model is None:
        return None, "Models not loaded yet, please check model path"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # try:
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Process condition image
        if cond_image is not None:
            cond_image = resize_with_aspect_ratio(cond_image, 1024)
            cond_tensor = process_image(cond_image)
            cond_tensor = cond_tensor.to(device)
            
            # VAE encoding
            vae_scale = 0.3611
            vae_shift = 0.1159
            cond_latent = (vae.encode(cond_tensor[None].to(dtype)).latent_dist.mode()[0] - vae_shift) * vae_scale
            cond_latent = cond_latent.float()
            cond = [[cond_latent]]
            if "object reference" in system_prompt:
                position_type = [["offset"]]
            else:
                position_type = [["aligned"]]
        else:
            raise ValueError("Condition image cannot be empty")
        
        # Full prompt
        full_prompt = system_prompt + prompt
        print(full_prompt)
        
        # Encode prompt
        cap_feats, cap_mask = encode_prompt([full_prompt] + [""], text_encoder, tokenizer)
        cap_mask = cap_mask.to(cap_feats.device)
        
        # Prepare model parameters
        w, h = cond_latent.shape[-2:] if cond_image is not None else (128, 128)
        z = torch.randn([1, 16, w, h], device=device).to(dtype)
        z = z.repeat(2, 1, 1, 1)
        cond = cond * 2
        position_type = position_type * 2

        model_kwargs = dict(
            cap_feats=cap_feats,
            cap_mask=cap_mask,
            cfg_scale=cfg_scale,
            cond=cond,
            position_type=position_type,
        )
        
        # Sampling
        sample_fn = sampler.sample_ode(
                            sampling_method=args.solver,
                            num_steps=args.num_sampling_steps,
                            atol=args.atol,
                            rtol=args.rtol,
                            reverse=args.reverse,
                            time_shifting_factor=t_shift
                        )
        samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
        
        # Decode generated image
        samples = vae.decode(samples / vae.config.scaling_factor + vae.config.shift_factor)[0]
        samples = (samples + 1.0) / 2.0
        samples.clamp_(0.0, 1.0)
        
        # Convert to PIL image
        output_image = to_pil_image(samples[0].float().cpu())
        
        return output_image, "Successfully generated image! \n System prompt: " + system_prompt + "\n Prompt: " + prompt
    
    # except Exception as e:
    #     return None, f"Error during generation: {str(e)}"

def create_demo():
    """Create Gradio demo interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# Lumina-Accessory Demo")
        
        with gr.Row():
            with gr.Column():
                cond_image = gr.Image(label="Condition Image", type="pil", format="jpeg")
                prompt = gr.Textbox(label="Text Prompt", lines=3, placeholder="Enter text describing the image...")
                system_prompt_dropdown = gr.Dropdown(
                    choices=list(SYSTEM_PROMPTS.keys()),
                    value="Text-to-Image",
                    label="System Prompt Type"
                )
                
                generate_condition = gr.Checkbox(label="Generate Condition Image (⚠️: If you upload a natural image and need to generate the condition image online.)", value=False)
                condition_type = gr.Dropdown(
                    choices=["Canny Edge", "Depth Map", "Brightness Map", "Palette Map"],
                    value="Canny Edge",
                    label="Condition Type",
                    visible=False
                )
                
                with gr.Row():
                    num_steps = gr.Slider(minimum=10, maximum=100, value=40, step=1, label="Inference Steps")
                    cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="Guidance Scale")
                
                with gr.Row():
                    t_shift = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="Time Shift Factor")
                    seed = gr.Number(label="Random Seed (leave empty for random)", precision=0, value=42)
                
                generate_btn = gr.Button("Generate Image")
                
            with gr.Column():
                with gr.Row():
                    condition_preview = gr.Image(label="Condition Preview", visible=True, format="jpeg")
                output_image = gr.Image(label="Generated Image", format="jpeg")
                output_message = gr.Textbox(label="Status Message")
        
        # set event handlers
        def get_system_prompt(choice):
            return SYSTEM_PROMPTS[choice]
            
        # show/hide condition type dropdown based on whether generate condition is checked
        generate_condition.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[generate_condition],
            outputs=[condition_type]
        )
        
        # preview condition image
        def preview_condition(img, cond_type):
            if img is None:
                return None
            
            if cond_type == "Canny Edge":
                cond_img = get_canny_edges(img)
            elif cond_type == "Depth Map":
                cond_img = get_depth_map(img)
            elif cond_type == "Brightness Map":
                cond_img = get_brightness_map(img)
            elif cond_type == "Palette Map":
                cond_img = get_palette_map(img)
            else:
                return None
                
            return cond_img
        
        # when condition type or input image changes, preview condition
        cond_image.change(
            fn=lambda img, cond_type, gen_cond: preview_condition(img, cond_type) if gen_cond else None,
            inputs=[cond_image, condition_type, generate_condition],
            outputs=[condition_preview]
        )
        
        condition_type.change(
            fn=lambda img, cond_type, gen_cond: preview_condition(img, cond_type) if gen_cond else None,
            inputs=[cond_image, condition_type, generate_condition],
            outputs=[condition_preview]
        )
        
        # generate image - first define this function
        def process_and_generate(img, txt, sys_prompt, gen_cond, cond_type, steps, scale, t_shift_val, seed_val):
            if img is None:
                return None, None, "Please upload a condition image first"
            
            # if generate condition, process input image
            if gen_cond:
                if cond_type == "Canny Edge":
                    cond_img = get_canny_edges(img)
                    sys_prompt_key = "Edge Condition"
                elif cond_type == "Depth Map":
                    cond_img = get_depth_map(img)
                    sys_prompt_key = "Depth Condition"
                elif cond_type == "Brightness Map":
                    cond_img = get_brightness_map(img)
                    sys_prompt_key = "Brightness Condition"
                elif cond_type == "Palette Map":
                    cond_img = get_palette_map(img)
                    sys_prompt_key = "Palette Condition"
                else:
                    return None, None, "Invalid Condition Type"
                
                system_prompt = SYSTEM_PROMPTS[sys_prompt_key]
                
                output_img, message = generate_image(
                    cond_img, txt, system_prompt, steps, scale, t_shift_val,
                    int(seed_val) if seed_val else None
                )
                
                return cond_img, output_img, message
            else:
                system_prompt = get_system_prompt(sys_prompt)
                output_img, message = generate_image(
                    img, txt, system_prompt, steps, scale, t_shift_val,
                    int(seed_val) if seed_val else None
                )
                
                return img, output_img, message
        
        generate_btn.click(
            fn=process_and_generate,
            inputs=[cond_image, prompt, system_prompt_dropdown, generate_condition, condition_type, 
                   num_steps, cfg_scale, t_shift, seed],
            outputs=[condition_preview, output_image, output_message]
        )
        
        # 修改示例区域 - 只显示按钮而不显示内容
        with gr.Row():
            gr.Markdown("## Examples")
            
        # 定义三个示例按钮
        with gr.Row():
            example1_btn = gr.Button("Example 1: Image Infilling")
            example2_btn = gr.Button("Example 2: Palette Condition")
            example3_btn = gr.Button("Example 3: Depth Condition")
        
        # 定义示例数据
        example_data = [
            [
                "examples/case_1_condition.jpg",  # condition image
                "A classical oil painting of a young woman dressed in a modern DARK BLACK leather jacket.", # prompt
                "Image Infilling",  # system prompt type
                False,  # generate condition
                "Canny Edge",  # condition type but not used here
                40,  # inference steps
                4.0,  # guidance scale
                6,  # time shift factor
                20,  # random seed
            ],
            [
                "examples/case_2_condition.jpg",
                "A still life photograph of a floral arrangement in a rustic, blue ceramic vase, centrally positioned on a round table draped with a delicate, white tablecloth. The bouquet features a mix of vibrant flowers, including large yellow roses, orange carnations, and smaller white blossoms, interspersed with green foliage and sprigs of orange buds. The vase is with the flowers extending upwards and outwards, creating a dynamic composition. In the background, hanging on the textured, beige wallpaper with a subtle floral pattern, is a traditional Chinese scroll featuring elegant calligraphy in classical Wenyanwen (文言文). The presence of the scroll adds a refined, cultural depth to the vintage setting. Soft, natural lighting casts gentle shadows, enhancing the textures of the vase and the lace. The overall atmosphere is serene and nostalgic, with a warm, muted color palette, medium depth of field, and a classic, timeless aesthetic.",
                "Palette Condition",
                False,
                "Canny Edge",
                40,
                4.0,
                6,
                20,
            ],
            [
                "examples/case_3_condition.jpg",
                "A contemplative photograph of a person with short brown hair, wearing a dark jacket, standing in the lower left foreground, facing away towards a field of tall, dried grasses. The grasses dominate the middle ground, their brown and beige tones contrasting with the dark jacket. The background features a cloudy, overcast sky with a soft, diffused light, creating a serene and introspective atmosphere. The composition is balanced with the person anchoring the lower left and the expansive sky occupying the upper half. The image has a muted color palette, emphasizing earthy tones and a sense of solitude. Photographic style, medium depth of field, natural lighting, soft focus, tranquil, introspective mood.",
                "Depth Condition",
                False,
                "Canny Edge",
                40,
                4.0,
                6,
                42,
            ]
        ]
        
        # 为每个按钮设置点击事件
        def load_example(example_idx):
            example = example_data[example_idx]
            return [
                example[0],  # cond_image
                example[1],  # prompt
                example[2],  # system_prompt_dropdown
                example[3],  # generate_condition
                example[4],  # condition_type
                example[5],  # num_steps
                example[6],  # cfg_scale
                example[7],  # t_shift
                example[8],  # seed
            ]
        
        example1_btn.click(
            fn=lambda: load_example(0),
            inputs=[],
            outputs=[
                cond_image, prompt, system_prompt_dropdown, 
                generate_condition, condition_type,
                num_steps, cfg_scale, t_shift, seed
            ]
        )
        
        example2_btn.click(
            fn=lambda: load_example(1),
            inputs=[],
            outputs=[
                cond_image, prompt, system_prompt_dropdown, 
                generate_condition, condition_type,
                num_steps, cfg_scale, t_shift, seed
            ]
        )
        
        example3_btn.click(
            fn=lambda: load_example(2),
            inputs=[],
            outputs=[
                cond_image, prompt, system_prompt_dropdown, 
                generate_condition, condition_type,
                num_steps, cfg_scale, t_shift, seed
            ]
        )
        
    return demo

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).",
    )
    group.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="the prediction model for the transport dynamics.",
    )
    group.add_argument(
        "--loss-weight",
        type=none_or_str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="the weighting of different components in the loss function, can be 'velocity' for dynamic modeling, 'likelihood' for statistical consistency, or None for no weighting.",
    )
    group.add_argument("--sample-eps", type=float, help="sampling in the transport model.")
    group.add_argument("--train-eps", type=float, help="training to stabilize the learning process.")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    group.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    group.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    group.add_argument(
        "--likelihood",
        action="store_true",
        help="Enable calculation of likelihood during the ODE solving process.",
    )



def none_or_str(value):
    if value == "None":
        return None
    return value


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    print("Parsing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--t_shift", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
        "(sample{_ema}.jpeg in the model checkpoint directory).",
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="prompts.txt",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="",
        nargs="+",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
    )
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="None",
        choices=["Time-aware", "None"],
    )
    parser.add_argument(
        "--system_type",
        type=str,
        default="real",
        # choices=["Time-aware", "None"],
    )
    parser.add_argument(
        "--scaling_watershed",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse", "sdxl", "flux"], default="flux"
    )
    parser.add_argument(
        "--text_encoder", type=str, nargs='+', default=['gemma'], help="List of text encoders to use (e.g., t5, clip, gemma)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Max length for text encoder."
    )
    parser.add_argument(
        "--use_parallel_attn",
        type=bool,
        default=False,
        help="Use parallel attention in the model.",
    )
    parser.add_argument(
        "--use_flash_attn",
        type=bool,
        default=True,
        help="Use Flash Attention in the model.",
    )
    parser.add_argument("--do_shift", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--training_type", type=str, default="full_model")
    parser.add_argument("--lora_rank", type=int, default=128, help="Rank for LoRA adaptation")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Scale for LoRA adaptation")

    parse_transport_args(parser)
    parse_ode_args(parser)

    parser.add_argument("--share", action="store_true", help="是否共享Gradio演示")

    args = parser.parse_known_args()[0]
    print(f"Arguments parsed: {args}")
    
    # Load models first
    load_models(args)
    
    print("Preparing to launch Gradio demo")
    demo = create_demo()
    print("Gradio demo created, preparing to launch")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=10090,
        share=False,
        max_threads=10
    )
    print("Gradio demo launched successfully")
