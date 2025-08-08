import gradio as gr
import spaces
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter, ImageOps
import numpy as np

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline


if gr.NO_RELOAD:
    MODELS = {
        "RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning",
    }

    config_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="config_promax.json",
    )

    config = ControlNetModel_Union.load_config(config_file)
    controlnet_model = ControlNetModel_Union.from_config(config)
    model_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="diffusion_pytorch_model_promax.safetensors",
    )
    state_dict = load_state_dict(model_file)
    model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
    )
    model.to(device="cuda", dtype=torch.float16)

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    ).to("cuda")

    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=torch.float16,
        vae=vae,
        controlnet=model,
        variant="fp16",
    ).to("cuda")

    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)


@spaces.GPU(duration=16)
def fill_image(image, model_selection, prompt):
    source = image["background"]
    mask = image["layers"][0]

    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = source.copy()
    cnet_image.paste(0, (0, 0), binary_mask)

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt, "cuda", True)

    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    ):
        yield image, cnet_image

    image = image.convert("RGBA")
    image = image.resize(cnet_image.size)
    cnet_image.paste(image, (0, 0), binary_mask)

    yield source, cnet_image


def blur_image(image, result, filter_size=16):
    # create mask
    alpha_channel = image["layers"][0].split()[3]

    # blur whole filled image
    filled_image = result[-1]
    blurred_image = Image.fromarray(filled_image)
    blurred_image = blurred_image.filter(ImageFilter.GaussianBlur(filter_size))

    # paste blurred part into the filled image
    blurred_image.paste(
        Image.fromarray(filled_image), mask=ImageOps.invert(alpha_channel)
    )

    return image["background"], blurred_image


def clear_result():
    return gr.update(value=None)


title = """<h1 align="center">Diffusers Image Fill</h1>
<div align="center">Draw the mask over the subject you want to erase or change.</div>
<div align="center">This space is a PoC made for the guide <a href='https://huggingface.co/blog/OzzyGT/diffusers-image-fill'>Diffusers Image Fill</a>.</div>
"""

with gr.Blocks() as demo:
    gr.HTML(title)

    run_button = gr.Button("Generate")

    with gr.Row(max_height=500):
        prompt = gr.Textbox(
            label="Prompt", value="A photo-realistic face, high-quality"
        )

        # prompt.change(update_prompt, inputs=[prompt])

    with gr.Row():
        input_image = gr.ImageMask(
            type="pil",
            label="Input Image",
            crop_size=(1024, 1024),
            canvas_size=(1024, 1024),
            layers=False,
            sources=["upload"],
            height=500,
        )

        result = ImageSlider(interactive=False, label="Generated Image", height=500)

        with gr.Column():
            with gr.Row(100):
                gauss_slider = gr.Slider(
                    5,
                    52,
                    value=16,
                    label="Gaussian Kernel Size",
                    info="Choose a value between 5 and 52. Higher values will result in increased blur.",
                    interactive=True,
                )

            with gr.Row(400):
                result_blurred = ImageSlider(
                    interactive=False, label="Blurred Image", height=400
                )

    model_selection = gr.Dropdown(
        choices=list(MODELS.keys()),
        value="RealVisXL V5.0 Lightning",
        label="Model",
    )

    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=fill_image,
        inputs=[input_image, model_selection, prompt],
        outputs=result,
    ).then(
        fn=blur_image,
        inputs=[input_image, result, gauss_slider],
        outputs=result_blurred,
    )

    gauss_slider.release(
        blur_image, inputs=[input_image, result, gauss_slider], outputs=result_blurred
    )


demo.launch(server_name="0.0.0.0")
