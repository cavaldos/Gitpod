import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-4.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    custom_pipeline="lpw_stable_diffusion_xl",
    add_watermarker=False
)
pipe.to('cuda')

prompt = "1girl, arima kana, oshi no ko, hoshimachi suisei, hoshimachi suisei \(1st costume\), cosplay, looking at viewer, smile, outdoors, night, v, masterpiece, high score, great score, absurdres"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=832,
    height=1216,
    guidance_scale=6,
    num_inference_steps=25
).images[0]

image.save("./arima_kana.png")
