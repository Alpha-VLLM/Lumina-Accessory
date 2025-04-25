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

#############################################################################
#                            Condition Generator                            #
#############################################################################
depth_pipeline = None

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

def get_masked_image(image):
    return image

def get_degraded_image(image):
    return image

def get_brightness_map(image):
    image = image.filter(ImageFilter.GaussianBlur(6))
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lower_thresh = 85 
    upper_thresh = 170 
    brightness_map = np.zeros_like(gray_image)
    brightness_map[gray_image <= lower_thresh] = 0  # Black
    brightness_map[(gray_image > lower_thresh) & (gray_image <= upper_thresh)] = 128  # Gray 
    brightness_map[gray_image > upper_thresh] = 255  # White
    brightness_map = Image.fromarray(brightness_map.astype(np.uint8))
    return brightness_map

def get_palette_map(image, num_colors=8):
    """Convert image to blocks of specified number of colors"""
    image = image.filter(ImageFilter.GaussianBlur(12))
    w, h = image.size
    # Resize image to reduce details
    small_img = image.resize((w // 32, h // 32), Image.Resampling.NEAREST)
    
    # Convert to numpy array for processing
    img_array = np.array(small_img)
    pixels = img_array.reshape(-1, 3)
    
    # Use K-means clustering to find main colors
    pixels = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Replace each pixel with its nearest cluster center color
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_array.shape)
    
    # Convert back to PIL image and resize to original size
    palette_map = Image.fromarray(quantized)
    palette_map = palette_map.resize((w, h), Image.Resampling.NEAREST)  # Use NEAREST to keep block boundaries clear
    
    return palette_map

def get_cond_image(image, task_type):
    if task_type == "Image Infilling":
        return get_masked_image(image)
    elif task_type == "Edge Condition":
        return get_canny_edges(image)
    elif task_type == "Depth Condition":
        return get_depth_map(image)
    elif task_type == "Brightness Condition":
        return get_brightness_map(image)
    elif task_type == "Palette Condition":
        return get_palette_map(image)
    elif task_type == "Image Restoration":
        return get_degraded_image(image)
    else:
        return None

def get_system_prompt(task_type):
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["Text-to-Image"])

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

#############################################################################
#                            Data item Processor                            #
#############################################################################

def resize_with_aspect_ratio(img, resolution=1024, divisible=64, aspect_ratio=None):
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
    
    # Ensure divisible
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

class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform
        self.special_format_set = set()
        self.cond_resolution = 1024
        self.img_resolution = 1024

    def text_image_pair_processor(self, data_item, task_type=None):
        task_types = ["Text-to-Image", "Image Infilling", "Image Restoration", "Edge Condition", 
                      "Depth Condition", "Brightness Condition", "Palette Condition"]
        sample_prob = [1, 1, 1, 1, 1, 1, 1]
        if task_type is None:
            task_type = random.choices(task_types, weights=sample_prob)[0]
        url = data_item["image_path"]
        try:
            image = Image.open(url)
            text = data_item["prompt"]
        except Exception as e:
            image = Image.new('RGB', (1024, 1024), color='black')
            text = "a black image"
            print(e)
            print("CONNOT OPEN", url)
        image = resize_with_aspect_ratio(image, self.img_resolution)

        if image.mode.upper() != "RGB":
            mode = image.mode.upper()
            try:
                image = image.convert("RGB")
            except Exception as e:
                raise NonRGBError()

        if task_type == "Text-to-Image":
            cond_images = []
        else:
            cond_images = [get_cond_image(image, task_type)]
        
        system_prompt = get_system_prompt(task_type)
        
        for i in range(len(cond_images)):
            if isinstance(cond_images[i], Image.Image) and cond_images[i].mode != 'RGB':
                cond_images[i] = cond_images[i].convert('RGB')

        image = self.image_transform(image)
        cond_images = [self.image_transform(cond_image) for cond_image in cond_images]
        text = system_prompt + text
        cond_images = [resize_with_aspect_ratio(cond_image, self.cond_resolution) for cond_image in cond_images]
        return image, text, cond_images

    def subject_driven_processor(self, data_item):
        image = Image.open(data_item["input_image"])
        cond = [Image.open(data_item["output_image"])]
        prompt = data_item["prompt"]
        text = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and an object reference. <Prompt Start> " + prompt
        image = self.image_transform(image)
        cond = [self.image_transform(cond_image) for cond_image in cond]
        image = resize_with_aspect_ratio(image, self.img_resolution)
        cond = [resize_with_aspect_ratio(cond_image, self.cond_resolution) for cond_image in cond]
        return image, text, cond

    def editing_processor(self, data_item):
        image = Image.open(data_item["input_image"])
        cond = [Image.open(data_item["output_image"])]
        prompt = data_item["prompt"]
        text = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on editing instructions and an original image. <Prompt Start> " + prompt
        image = self.image_transform(image)
        cond = [self.image_transform(cond_image) for cond_image in cond]
        image = resize_with_aspect_ratio(image, self.img_resolution)
        cond = [resize_with_aspect_ratio(cond_image, self.cond_resolution) for cond_image in cond]
        return image, text, cond

    def process_item(self, data_item, group_name=None, training_mode=False):
        return data_item, group_name

def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
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

def main(args, rank, master_port):
    # Setup PyTorch:
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = rank % torch.cuda.device_count()
    device_str = f"cuda:{device}"
    
    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    if dist.get_rank() == 0:
        print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    if dist.get_rank() == 0:
        print(f"Creating text encoder: {args.text_encoder}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.padding_side = "right"

    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2-2b", torch_dtype=dtype, device_map="cuda"
    ).eval()
    cap_feat_dim = text_encoder.config.hidden_size
    
    if args.vae == "flux":
        vae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device_str)
    elif args.vae != "sdxl":
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
            if args.local_diffusers_model_root is None
            else os.path.join(args.local_diffusers_model_root, f"stabilityai/sd-vae-ft-{args.vae}")
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.requires_grad_(False)

    if dist.get_rank() == 0:
        print(f"Creating DiT: {train_args.model}")
    model = models.__dict__[train_args.model](
        in_channels=16,
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    if args.training_type == "lora":
        print(f"Applying LoRA with rank {args.lora_rank} and scale {args.lora_scale}")
        replace_linear_with_lora(model, max_rank=args.lora_rank, scale=args.lora_scale)
    model.eval().to("cuda", dtype=dtype)
    
    if args.debug == False:
        if args.ema:
            print("Loading ema model.")
        ckpt = torch.load(
            os.path.join(
                args.ckpt,
                f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.pth",
            )
        )
        model.load_state_dict(ckpt, strict=True)
    
    # begin sampler``
    if args.solver == "dpm":
        transport = create_transport(
            "Linear",
            "velocity",
        )
        sampler = Sampler(transport)
    else:
        transport = create_transport(
            args.path_type,
            args.prediction,
            args.loss_weight,
            args.train_eps,
            args.sample_eps,
        )
        sampler = Sampler(transport)

    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, "images"), exist_ok=True)
        print(f"Saving .jpeg samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, "data.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.loads(f.read())
        collected_id = []
    else:
        info = []
        collected_id = []

    if args.caption_path.endswith("json"):
        with open(args.caption_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    elif args.caption_path.endswith("jsonl"):
        data = []
        with open(args.caption_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line))
    else:
        data = []
        with open(args.caption_path, "r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    data.append(line.strip())
    neg_cap = ""
    total = len(info)
    resolution = args.resolution

    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    item_processor = T2IItemProcessor(image_transform)

    vae_scale = {
        "sdxl": 0.13025,
        "sd3": 1.5305,
        "ema": 0.18215,
        "mse": 0.18215,
        "cogvideox": 1.15258426,
        "flux": 0.3611,
    }["flux"]
    vae_shift = {
        "sdxl": 0.0,
        "sd3": 0.0609,
        "ema": 0.0,
        "mse": 0.0,
        "cogvideox": 0.0,
        "flux": 0.1159,
    }["flux"]
    
    with torch.autocast("cuda", dtype):
        for res in resolution:
            for idx, item in tqdm(enumerate(data)):
                if int(args.seed) != 0:
                    torch.random.manual_seed(int(args.seed))

                sample_id = f'{idx}_{res.split(":")[-1]}'
                
                save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}_{args.task_type}.jpeg"
                if os.path.exists(save_path):
                    continue

                res_cat, resolution = res.split(":")
                res_cat = int(res_cat)
                item_processor.img_resolution = res_cat
                item_processor.cond_resolution = res_cat

                if args.task_type in ["Text-to-Image", "Image Infilling", "Image Restoration", 
                                "Edge Condition", "Depth Condition", "Brightness Condition", 
                                "Palette Condition", "Human Keypoint Condition"]:
                    image, cap, cond = item_processor.text_image_pair_processor(item, args.task_type)
                    position_type = [["aligned"]]
                elif args.task_type == "Subject-driven":
                    image, cap, cond = item_processor.subject_driven_processor(item)
                    position_type = [["offset"]]
                elif args.task_type in ["Image Relighting", "Image Editing"]:
                    image, cap, cond = item_processor.editing_processor(item)
                    position_type = [["aligned"]]
                else:
                    # 默认使用text_image_pair_processor
                    image, cap, cond = item_processor.text_image_pair_processor(item, args.task_type)
                    position_type = [["aligned"]]

                neg_cap = ""

                caps_list = [cap]
                original_cond = cond
                cond = [cond]
                
                n = len(caps_list)
                if cond[0] != []:
                    w, h = cond[0][0].shape[-2:]
                else:
                    w, h = resolution.split("x")
                    w, h = int(w), int(h)
                latent_w, latent_h = w // 8, h // 8
                z = torch.randn([1, 16, latent_w, latent_h], device="cuda").to(dtype)
                z = z.repeat(n * 2, 1, 1, 1)

                cond = [[cond_img.to("cuda", non_blocking=True) for cond_img in cond_imgs] for cond_imgs in cond]
                for i in range(len(cond)):
                    cond[i] = [(vae.encode(cond_img[None].bfloat16()).latent_dist.mode()[0] - vae_shift) * vae_scale for cond_img in cond[i]]
                    cond[i] = [cond_img.float() for cond_img in cond[i]]
                cond = cond * 2
                position_type = position_type * 2

                with torch.no_grad():
                    cap_feats, cap_mask = encode_prompt([cap] + [neg_cap], text_encoder, tokenizer, 0.0)
        
                cap_mask = cap_mask.to(cap_feats.device)
                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=cap_mask,
                    cfg_scale=args.cfg_scale,
                    cond=cond,
                    position_type=position_type,
                )

                start_time = time.perf_counter()
                if args.solver == "dpm":
                    sample_fn = sampler.sample_dpm(
                        model.forward_with_cfg,
                        model_kwargs=model_kwargs,
                    )
                    samples = sample_fn(z, steps=args.num_sampling_steps, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=args.time_shifting_factor)
                else:
                    sample_fn = sampler.sample_ode(
                        sampling_method=args.solver,
                        num_steps=args.num_sampling_steps,
                        atol=args.atol,
                        rtol=args.rtol,
                        reverse=args.reverse,
                        time_shifting_factor=args.t_shift
                    )
                    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]

                end_time = time.perf_counter()
                samples = samples[:1]
                samples = vae.decode(samples / vae.config.scaling_factor + vae.config.shift_factor)[0]
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)
                print("sample times:", end_time-start_time)

                # Save samples to disk as individual .jpeg files
                for i, (sample, cap) in enumerate(zip(samples, caps_list)):
                    img = to_pil_image(sample.float())
                    save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}_{args.task_type}.jpeg"
                    img.save(save_path)
                    info.append(
                        {
                            "caption": cap,
                            "image_url": f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}_{args.task_type}.jpeg",
                            "resolution": f"res: {resolution}\ntime_shift: {args.time_shifting_factor}",
                            "solver": args.solver,
                            "num_sampling_steps": args.num_sampling_steps,
                        }
                    )
                
                for cond_idx, cond_img in enumerate(original_cond):
                    cond_img = (cond_img.float() + 1.0) / 2.0  
                    cond_img.clamp_(0.0, 1.0)  
                    cond_pil = to_pil_image(cond_img)
                    save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}_{args.task_type}_cond_{cond_idx}.jpeg"
                    cond_pil.save(save_path)

                with open(info_path, "w") as f:
                    f.write(json.dumps(info))

                total += len(samples)
                dist.barrier()
    end_time = time.time()
    print("sample times:", end_time-start_time)
    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
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
        default=["1024:1024x1024"],
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
    parser.add_argument("--task_type", type=str, default="Text-to-Image", 
                        choices=["Text-to-Image", "Image Infilling", "Image Restoration", 
                                "Edge Condition", "Depth Condition", "Brightness Condition", 
                                "Palette Condition", "Human Keypoint Condition", 
                                "Subject-driven", "Image Relighting", "Image Editing"])

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)
