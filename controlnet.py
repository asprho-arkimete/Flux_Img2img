import torch
from PIL import Image
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import qfloat8, quantize, freeze

# === Configurazioni iniziali ===
model_id = "black-forest-labs/FLUX.1-dev"
model_id_nsfw = "trongg/FLUX.1-dev_nsfw_FLUXTASTIC-v3.0"  # Modello NSFW compatibile
lorapath = "./Lora/flux hairy-pussy.safetensors"
image_path = "./J cortess/j2.jpg"
dtype = torch.bfloat16  # Precisione base usata per il modello


# === Ridimensiona immagine di input a 512px sul lato lungo ===
def ridimensiona_a_512(img):
    w, h = img.size
    if w >= h:
        new_h = (512 * h) // w
        img = img.resize((512, new_h), Image.BICUBIC)
    else:
        new_w = (512 * w) // h
        img = img.resize((new_w, 512), Image.BICUBIC)
    return img


# === Caricamento della pipeline principale ===
print("🔄 Caricamento della pipeline...")
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)


# === Caricamento e quantizzazione del transformer separatamente ===
print("🔄 Caricamento e quantizzazione del transformer...")
transformer = FluxTransformer2DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=dtype
)
quantize(transformer, weights=qfloat8)
freeze(transformer)

# === Caricamento e quantizzazione di text_encoder_2 separatamente ===
print("🔄 Caricamento e quantizzazione di text_encoder_2...")
text_encoder_2 = T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder_2", torch_dtype=dtype
)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)



# === Caricamento della LoRA (DOPO la quantizzazione!) ===
print("📎 Caricamento della LoRA...")
import os
pipe.load_lora_weights(lorapath, weight_name=os.path.basename(lorapath), adapter_name=os.path.basename(lorapath)[0])
pipe.set_adapters(os.path.basename(lorapath)[0], adapter_weights=1.0)
print("✅ Adapters attivi:", pipe.get_active_adapters())

print("riaplico i modelli trrasformers e text_encoder quantificati dopo il Lora")
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2

# === Caricamento dell'IP-Adapter ===
print("📎 Caricamento dell'IP-Adapter...")
pipe.load_ip_adapter(
    "XLabs-AI/flux-ip-adapter",
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
)
pipe.set_ip_adapter_scale(0.5)

# === Abilita offload automatico per risparmiare VRAM ===
pipe.enable_model_cpu_offload()


# === Prepara immagine di riferimento ===
print("🖼️ Preparazione immagine di input...")
image = Image.open(image_path).convert("RGB")
image = ridimensiona_a_512(image)


# === Prompt per la generazione ===
prompt = (
    "a nude Latina woman sitting on a chair with legs apart. Photorealistic, ultra-high detail, soft natural lighting. "
    "Focus on lower body anatomy, natural pubic hair, explicit but artistic view. "
    "Black hair in a chignon, brown skin, soft expression, 8K, masterpiece."
)

# === Generazione dell'immagine ===
print("🎨 Generazione dell'immagine...")
generator = torch.Generator(device="cuda").manual_seed(42)  # Usa un seed fisso per riproducibilità


images = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    guidance_scale=6.0,
    num_inference_steps=50,
    generator=generator,
    ip_adapter_image=image,
    max_sequence_length=512,
).images


# === Salvataggio risultato ===
print("💾 Salvataggio immagine...")
# Evita problemi di valori fuori scala usando .clamp()
images[0] = images[0].convert("RGB")  # Forza RGB se necessario
images[0].save("results.jpg")
print("✅ Immagine salvata come 'results.jpg'")