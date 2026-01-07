
### 1.   INPUT PROMPT HERE   (g_clip_prompt)
g_clip_prompt = "Futuristic glass-domed greenhouse set against a sprawling cityscape where a robotic gardener stands among rows of organized plantes, the greenhouse's clean lines contrasting with towering skyscrapers in the background. the rows of plants have delicate vines curling around each container. The robot’s metallic surface shows light pitting and corrosion on its aging chassis, while tiny dewdrops cling to leaves, glinting faintly in the filtered light. "
l_clip_prompt = g_clip_prompt  
# g_clip is main prompt, l_clip is 'style prompt'    Usually these can be the same.
#        --- flux uses these two text encoders (g and l) in parallel. the clip-l model is smaller and designed to use tags (comma separated words, phrases), while the clip-g model is more sophisticated and can interpret plain english
#        --- prompt suggestion: use ,, to end a sentence, and only use a period . to separate sections of the prompt --  periods that end sentences can act as dividers for concepts.  If things interact (steam rising from coffee), then mention them in the same 'sentence'. I like to use dual commas ,, to end a sentence where I don't want to partition off the concepts from each other
# for longer g_clip prompts, you may want to create a shorter l_clip prompt that is comma-separated tags "futuristic, glass-dome, greenhouse, sprawling cityscape, gardener robot, rows of plants, delicate vines"

### 2. CHOOSE RESOLUTION, Suggested are (wxh) 832x1216, 1024x1024, 1216x832
width = 832
height = 1216

### 3. SET NUMBER OF STEPS AND GUIDANCE STRENGTH (suggest 23 steps, 2 guidance)
num_inference_steps = 23 # use 20-50
guidance_scale = 2 # use 1.8-7, usually 2-3.5

### 4. IDENTIFY LORA MODELS TO USE - Flux is very capable without using loras, you can leave this alone
# List of LoRA paths and their corresponding weights (./ for realtive path, / for absolute  in linux)
# weight is 1.0 by default, suggested range is 0.2 - 1.5 for most loras, and negative ranges can also have a beneficial effect
lora_configs = [   ## remove the # from one of these lines and put in your lora path
    #{"path": "path-to-lora-file-with-extension", "weight": 1.0 },
    #{"path": "/home/aikenyon/accel/lora/flux-photo-transparent_Alpha_10.safetensors", "weight": 1.0 },
    #{"path": "./lora/insane.safetensors", "weight": -0.5},  ## this lora adds unrealistic detail with a positive weight, but at -0.5 it removes the excess styling to reveal heightened realism
    # Add more LoRA configuration lines as needed 
]

### 5. set image count and output path - you can hit ctrl+c to cancel image generation
img_count = 4
output_folder_path = "./output"  # "./output" is just a folder that will be created automatically in this script's location

### 6. Set seed for reproducibility - setting randomize_seed overrides assigning a seed
seed = 42
randomize_seed = False



####  This script will automatically download flux dev and save to a .cache folder in your user profile.
####  IT WILL TAKE UP 30GB OF HDD SPACE

### You can probably change this to other models hosted on huggingface if they are set up to support remote  use
#USE FOR HF MODEL LOADING - automatically downloads model to cache
model_path = "black-forest-labs/FLUX.1-dev"
model_name = "FLUX.1-dev"





#################################################  code beyond here is not meant to be altered and may contain some dead or experimental code ####################################################
import random
from diffusers import  StableDiffusionXLImg2ImgPipeline,AutoencoderTiny,FluxPipeline
from torch import autocast
import torch
from pathlib import Path
import os
from PIL import Image, PngImagePlugin
from itertools import combinations
from transformers import AutoModelForCausalLM  
import math
import torch

torch.backends.cuda.matmul.allow_tf32 = True

#flux doesn't use negative prompts
g_clip_negprompt = ""
l_clip_negprompt = ""



#stage_1_strength = 2
#essential even for 24gb vram if using flux dev 16 bit
enable_sequential_cpu_offload = True

increment_seed = True

pair_test = False
open_image_on_gen = False
test_loras = False # for all loras in test path, iterate through each one per generation
test_lora_weight = 1.0
test_lora_path = "/home/aikenyon/comfyui-linux/ComfyUI/models/loras"


dtype = torch.bfloat16 # torch.bfloat16 / torch.float16 / torch.float32



def add_stable_diffusion_metadata(png_path, metadata):
    # Open the existing image
    with Image.open(png_path) as img:
        # Create a new PngInfo object
        pnginfo = PngImagePlugin.PngInfo()
        # Add the metadata to the PngInfo object
        pnginfo.add_text("parameters", metadata)
        # Save the image with the new metadata
        img.save(png_path, "PNG", pnginfo=pnginfo)

test_lora_configs = []
test_lora_count = 0
current_Test_lora = 1

if test_loras:
    for root, _, files in os.walk(test_lora_path):
        for file in files:
            if file.endswith(".safetensors"):
                test_lora_configs.append({
                    "path": os.path.join(root, file),
                    "weight": test_lora_weight
                })

    print(test_lora_configs)


def nearest_multiple_of_64(x):
    # Round the float to the nearest integer
    rounded = round(x)
    # Find the nearest multiple of 8
    nearest = round(rounded / 64) * 64    
    return nearest

def scale_to_nearest_multiple(image, scale, multiple, rounding_fraction=0.5):
    original_width, original_height = image.size
    original_ratio = original_width / original_height

    # Scale dimensions
    scaled_width = int(original_width * scale)
    scaled_height = int(original_height * scale)

    # Function to round to nearest multiple
    def round_to_multiple(value, base, fraction):
        return math.ceil((value + base * fraction) / base) * base

    # Find nearest multiples
    target_width = round_to_multiple(scaled_width, multiple, rounding_fraction)
    target_height = round_to_multiple(scaled_height, multiple, rounding_fraction)

    # Determine which dimension to fit to
    if target_width / original_ratio <= target_height:
        # Fit to width
        final_width = target_width
        final_height = int(final_width / original_ratio)
        final_height = (final_height // multiple) * multiple  # Adjust height to nearest smaller multiple
    else:
        # Fit to height
        final_height = target_height
        final_width = int(final_height * original_ratio)
        final_width = (final_width // multiple) * multiple  # Adjust width to nearest smaller multiple

    # Resize image
    resized_image = image.resize((final_width, final_height), Image.LANCZOS)

    # Crop if necessary
    if final_width > target_width or final_height > target_height:
        left = (final_width - target_width) // 2
        top = (final_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        resized_image = resized_image.crop((left, top, right, bottom))

    return resized_image

def truncate_filename(filename, max_length=100):
    name, ext = os.path.splitext(filename)
    if len(name) + len(ext) <= max_length:
        return filename
    return name[:max_length - len(ext)] + ext

def generate_lora_pairs(lora_configs):
    lora_names = [Path(config["path"]).stem for config in lora_configs]
    return list(combinations(lora_names, 2))

if pair_test:
    lora_pairs = generate_lora_pairs(lora_configs)
    print(f"Generated {len(lora_pairs)} LoRA pairs for testing")


device = "cuda"
#torch.cuda.set_device(1)
#device = "cuda" if torch.cuda.is_available() else "cpu"


# scheduler = EulerAncestralDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
#scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
#scheduler2 = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
#can't define flux scheduler yet in diffusers

#create seed generator/container
generator = torch.Generator(device).manual_seed(seed)
generator2 = torch.Generator(device).manual_seed(seed)


def init_pipeline():
    # Initialize the SDXL pipeline from the SafeTensors file
    #pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)  #original pipe load before supporting low vram

    pipe = FluxPipeline.from_pretrained(
        model_path, 
        torch_dtype=dtype)    

    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once
    return pipe

pipe = init_pipeline()
pipe2 = None

def load_lora_weights_cumulatively(pipe, lora_configs):
    lora_load_success = False
    # Unload all current LoRA weights
    pipe.unload_lora_weights()
    print("Cleared all existing LoRA weights")
    try:
        for config in lora_configs:
            lpath = Path(config["path"])
            if not os.path.isfile(lpath):
                print(f"cannot find: {lpath}, skipping lora")
                lora_configs.remove(config)
                continue
            weight = float(config["weight"])
            try:
                float(weight)                
            except ValueError as e:
                print(f"error retrieving/parsing lora weight: {e}, skipping lora")
                lora_config.remove(config)
                continue
            adapter_name = lpath.stem
            print(f"Loading LoRA: {lpath} with weight: {weight}")
            
            pipe.load_lora_weights(lpath, weight_name="pytorch_lora_weights.safetensors", adapter_name=adapter_name)

        adapter_names = [Path(config["path"]).stem for config in lora_configs]
        adapter_weights = [float(config["weight"]) for config in lora_configs]

        print("Activating LoRAs with weights")
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

        print("All LoRA weights applied")
        lora_load_success = True
    except ValueError as e:
        print(f"attempt to load loras failed: {e}")
        lora_load_success = False

    return lora_load_success

# load_lora_configs(lora_configs)
if not test_loras and not pair_test and len(lora_configs) > 0:
    if not load_lora_weights_cumulatively(pipe, lora_configs):
        exit()

'''
#FOR SDXL - FOR FLUX, UNET IS CALLED TRANSFORMER - WON'T WORK FOR LORA TESTING UNTIL THIS IS CHANGED OVER AND TESTED
original_params = {
'unet': {name: param.clone() for name, param in pipe.unet.named_parameters() if 'lora' not in name},
'text_encoder': {name: param.clone() for name, param in pipe.text_encoder.named_parameters() if 'lora' not in name},
'text_encoder_2': {name: param.clone() for name, param in pipe.text_encoder_2.named_parameters() if 'lora' not in name}
}


def reset_models_to_original(pipeline):
    with torch.no_grad():
        for name, param in pipeline.unet.named_parameters():
            if name in original_params['unet']:
                param.copy_(original_params['unet'][name])
        for name, param in pipeline.text_encoder.named_parameters():
            if name in original_params['text_encoder']:
                param.copy_(original_params['text_encoder'][name])
        for name, param in pipeline.text_encoder_2.named_parameters():
            if name in original_params['text_encoder_2']:
                param.copy_(original_params['text_encoder_2'][name])
    pipeline.unet.set_attn_processor(AttnProcessor2_0())  
'''


# Function to increment file name
def increment_file_name(directory, file_name):
    path = Path(directory) / file_name
    stem = path.stem
    suffix = path.suffix
    counter = 1

    while path.exists():
        path = Path(directory) / f"{stem}_{counter}{suffix}"
        counter += 1
    
    return path.name  # Return just the file name, not the full path

img_counter = 1

lora_data_string = ""
if pair_test and lora_pairs:
    img_count = len(lora_pairs)
    print(f"changed img_count to number of lora pairs: {img_count}")
elif test_loras:
    img_count = len(test_lora_configs)
    print(f"changed img_count to number of loras: {img_count}")


curr_lora_configs = lora_configs

while img_counter <= img_count:
    print(f"Generating image {img_counter}/{img_count}")
    error_state = False
    new_lora_configs = None

    if pair_test and lora_pairs:
        curr_pair = lora_pairs.pop(0)
        new_lora_configs = [
            config for config in lora_configs
            if Path(config["path"]).stem in curr_pair
        ]
        print(f"Testing LoRA pair: {curr_pair}")
    elif test_loras:
        new_lora_configs = [test_lora_configs.pop()]
        new_lora_configs.extend(lora_configs)
    else:
        new_lora_configs = lora_configs

    
    # Only reload LoRA weights if the configuration has changed
    if new_lora_configs != curr_lora_configs or pipe is None:
        if pipe is None:
            pipe = init_pipeline()

        if not load_lora_weights_cumulatively(pipe, new_lora_configs):
            error_state = True
        curr_lora_configs = new_lora_configs
        print("Reloaded LoRA weights due to configuration change.")

    '''
    # Generate a latent image # NOT FOR FLUX ATM
    latent_batch_size = 1
    latent_color_channels = 3
    latent_shape = (latent_batch_size, latent_color_channels, height, width)  # Shape of the initial image
    
    #latent_image = torch.randn(latent_shape, device=device, dtype=torch.float16)
        #FLUX DIDN'T LIKE THIS INPUT, PROVIDING WIDTH AND HEIGHT DIRECTLY TO PIPE #
    # latent_image = torch.randn(latent_shape, device=device, dtype=dtype)

  
    ##FLUX INFERENCE FROM WORKING SCRIPT FOR REFERENCE
    image = pipe(
        prompt=clip_prompt,  # Your summarized, tag-based CLIP prompt
        prompt_2=t5_prompt,  # Your detailed T5 prompt
        guidance_scale=3.5,
        output_type="pil",
        num_inference_steps=20,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    file_path = get_incremented_filename("image.png")
    image.save(file_path)
    '''
    # Generate the image
    if not error_state:
        with autocast(device):
            images = pipe(                
                #image=latent_image,
                width=width,
                height=height,
                prompt=l_clip_prompt,
                prompt_2=g_clip_prompt,
                #negative_prompt=g_clip_negprompt,
                #negative_prompt_2=l_clip_negprompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=512,
                #strength=stage_1_strength
            ).images

        file_name_with_info = f"{model_name}_cfg-{guidance_scale}_steps-{num_inference_steps}"

        # Add LoRA info to the file name
        lora_info_string = ""
        for lora_config in curr_lora_configs:
            lora_path = lora_config["path"]
            lora_weight = lora_config["weight"]
            lora_name_stem = Path(lora_path).stem
            lora_info_string += f"_{lora_name_stem}-{lora_weight}"
        file_name_with_info += lora_info_string
        lora_data_string += lora_info_string
        
        file_name_with_info = truncate_filename(file_name_with_info, 100)

        # Save the image with an incremented file name
        output_file_name = increment_file_name(output_folder_path, f"{file_name_with_info}.png")
        output_path = os.path.join(output_folder_path, output_file_name)


        images[0].save(output_path)
        print(f"Image saved to {output_path}")

        metadata = (
        f"prompt: g_clip: {g_clip_prompt} l_clip: {l_clip_prompt} \n"
        f"negative_prompt: g_clip neg: {g_clip_negprompt} l_clip neg: {l_clip_negprompt}\n"
        f"steps: {num_inference_steps}\n"
        f"cfg_scale: {guidance_scale}\n"
        f"seed: {seed}\n"
        f"model: {model_name}\n"
        #f"sampler: {sampler_name}\n"
        f"sampler: FlowMatchEulerDiscrete\n"
        f"lora: {lora_data_string}\n"
        )
        add_stable_diffusion_metadata(output_path, metadata)
        
        


    if increment_seed:
        seed += 1
    elif randomize_seed:
        seed = random.randint(1, 4294,967295)

    img_counter += 1

    if open_image_on_gen:
        os.startfile(output_path)



exit()


# Generate the image
with autocast(device):
    images = pipe(
        prompt=g_clip_prompt,
        prompt_2=l_clip_prompt,
        height=height,																
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images

# Save the image with an incremented file name
output_path = increment_file_name("output.png")
images[0].save(output_path)
print(f"Image saved to {output_path}")
os.startfile(output_path)

