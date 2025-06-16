import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

import torch
from diffusers import FluxTransformer2DModel, FluxImg2ImgPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize

# Percorsi
path_img1 = "./images_fill/kiss1.jpg"
path_mask2 = "./images_fill/mask_preset.jpg"
path_reference = "./images_fill/faceNicole.jpg"

# Funzione per il resize
def resize_image(image: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = image.size
    if w >= h:
        new_w = max_side
        new_h = int(h * (max_side / w))
    else:
        new_h = max_side
        new_w = int(w * (max_side / h))
    return image.resize((new_w, new_h), resample=Image.LANCZOS)

# Caricamento e resize immagini
image = resize_image(Image.open(path_img1).convert("RGB"))
mask_image = resize_image(Image.open(path_mask2).convert("RGB"))
reference_image = resize_image(Image.open(path_reference).convert("RGB"))

print(f"Dimensioni dopo resize - Immagine: {image.size}, Maschera: {mask_image.size}, Ref: {reference_image.size}")

# Funzione per creare immagine di controllo per inpainting
def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

control_image = make_inpaint_condition(image, mask_image)

# Caricamento ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", 
    torch_dtype=torch.float16
)

# Caricamento del modello custom
loadmodel = ".//modelli//absolutereality181_arPM7Inpaintv10.safetensors"
pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
    loadmodel,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Caricamento LoRA
pipe.load_lora_weights(
    "Lora",
    weight_name="tongue28.safetensors",
    adapter_name="tongue"
)
pipe.set_adapters("tongue", adapter_weights=0.7)

# IMPORTANTE: Spostare su CUDA PRIMA di caricare IP-Adapter
pipe.to('cuda')

# Impostazioni pipeline
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# RIMUOVI questa riga che causa il conflitto:
# pipe.enable_model_cpu_offload()

# Carica IP-Adapter DOPO aver spostato il modello su CUDA
pipe.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="models", 
    weight_name="ip-adapter-full-face_sd15.bin",
    torch_dtype=torch.float16
)

pipe.set_ip_adapter_scale(0.6)

# Prompt
prompt = "a 24 year old girl with hair blonde and eyes blue, and a 13 year old girl with hair blonde and eyes blue, French kiss with tongues"

negative = """(cartoon) (monochrome) (asian) (overexposed) ((penis)) (out of focus) (blurry) (deformed mouth) (deformed pupils) 
(disfigured) (blue tongue) (extra limbs) (extra fingers) (Deformed) (plastic skin) (mutilated) (lowres) ((dark)) (boring) 
(lowpoly) (CG) (3d) (blurry) (duplicate) (watermark) (label) (signature) (frames) (text) (closed mouth:0.6) (eyes closed) 
(yellow teeth:0.9) 4k (long tongue:0.5) (closeup:0.1)"""

# Seed generator
generator = torch.Generator(device="cuda").manual_seed(42)

# Generazione immagine
result = pipe(
    prompt=prompt,
    negative_prompt=negative,
    num_inference_steps=100,
    generator=generator,
    eta=1.0,
    image=image,
    mask_image=mask_image,
    control_image=control_image,
    ip_adapter_image=reference_image,
    guidance_scale=7.5,
    strength=0.4,
).images[0]
result.save("images_fill/output.jpg")

def refine_flux(prompt, result, steps, modifica, cfg):
    """
    Funzione per il refinement con Flux
    Args:
        prompt: Prompt di testo
        result: Immagine PIL di input
        steps: Numero di inference steps
        modifica: Strength (0.0-1.0)
        cfg: Guidance scale
    Returns:
        PIL Image refined
    """
    try:
        # LIBERAZIONE COMPLETA VRAM - Rimozione primo pipeline
        print("🗑️ Liberazione VRAM dal primo pipeline...")
        # Verifica se le variabili esistono prima di eliminarle
        import gc
        globals_to_delete = ['pipe', 'controlnet']
        for var_name in globals_to_delete:
            if var_name in globals():
                del globals()[var_name]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Attende che tutte le operazioni CUDA finiscano
        print("✅ VRAM liberata completamente")

        bfl_repo = "black-forest-labs/FLUX.1-dev"
        dtype = torch.bfloat16

        # Ottimizzazioni VRAM - Caricamento e quantizzazione transformer
        print("🔄 Caricamento e quantizzazione Transformer...")
        transformer = FluxTransformer2DModel.from_single_file(
            ".//modelli//jibMixFlux_v85Consisteight.safetensors", 
            torch_dtype=dtype
        )
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        # Ottimizzazioni VRAM - Caricamento e quantizzazione text encoder
        print("🔄 Caricamento e quantizzazione Text Encoder...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            bfl_repo, 
            subfolder="text_encoder_2", 
            torch_dtype=dtype
        )
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        # Caricamento CLIP text encoder e quantizzazione
        print("🔄 Caricamento e quantizzazione CLIP Text Encoder...")
        text_encoder = CLIPTextModel.from_pretrained(
            bfl_repo, 
            subfolder="text_encoder", 
            torch_dtype=dtype
        )
        quantize(text_encoder, weights=qfloat8)
        freeze(text_encoder)

        # Creazione pipeline con componenti quantizzati
        print("🔄 Creazione pipeline Flux...")
        pipeflux = FluxImg2ImgPipeline.from_pretrained(
            bfl_repo, 
            transformer=None, 
            text_encoder=None,
            text_encoder_2=None, 
            torch_dtype=dtype
        )

        # Assegnazione componenti quantizzati
        pipeflux.transformer = transformer
        pipeflux.text_encoder = text_encoder
        pipeflux.text_encoder_2 = text_encoder_2

        # Ottimizzazioni VRAM per pipeline Flux
        print("🔄 Applicazione ottimizzazioni VRAM...")
        # Solo model_cpu_offload per compatibilità con quantizzazione
        pipeflux.enable_model_cpu_offload()

        # Ottimizzazioni VAE per risparmio memoria
        if hasattr(pipeflux.vae, 'enable_tiling'):
            pipeflux.vae.enable_tiling()
        if hasattr(pipeflux.vae, 'enable_slicing'):
            pipeflux.vae.enable_slicing()

        # Pulizia memoria CUDA prima della generazione
        torch.cuda.empty_cache()

        print("🎨 Generazione immagine refinement con Flux...")
        wres, hres = result.size

        imageout = pipeflux(
            prompt=prompt,
            image=result,
            width=wres,
            height=hres,
            num_inference_steps=int(steps),
            strength=float(modifica),
            guidance_scale=float(cfg),
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # Pulizia finale memoria e liberazione pipeline Flux
        print("🗑️ Liberazione finale VRAM...")
        del pipeflux
        del transformer
        del text_encoder
        del text_encoder_2
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return imageout
        
    except Exception as e:
        print(f"❌ Errore durante il refinement Flux: {e}")
        # Pulizia di emergenza in caso di errore
        torch.cuda.empty_cache()
        return result  # Ritorna l'immagine originale in caso di errore

# Salvataggio output
imageout = refine_flux(prompt=prompt, result=result, steps=25, modifica=0.85, cfg=6.0)
imageout.save("images_fill/output_refine.jpg")
print("✅ Immagine refinement salvata in: images_fill/output_refine.jpg")
print("✅ Processo completato - VRAM completamente liberata!")