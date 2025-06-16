import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import image_frame

##https://huggingface.co/spaces/emirhanbilgic/segment-hair

# Per compatibilità con mediapipe
_Image = image_module.Image
_ImageFormat = image_frame.ImageFormat

# Colori maschera
MASK_COLOR = (255, 255, 255, 255)  # capelli bianchi
BG_COLOR = (0, 0, 0, 255)          # sfondo nero

# Caricamento modello .tflite
base_options = python.BaseOptions(model_asset_path='emirhan.tflite')
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    output_category_mask=True
)

# Funzione principale
def segment_hair(image):
    # Converti l'immagine in RGBA e imposta trasparenza
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    rgba_image[:, :, 3] = 0

    # Crea oggetto immagine per MediaPipe
    mp_image = _Image(image_format=_ImageFormat.SRGBA, data=rgba_image)

    # Segmentazione capelli
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image)
        category_mask = result.category_mask.numpy_view()

    # Maschera binaria (bianco capelli, nero sfondo)
    mask = (category_mask > 0.2).astype(np.uint8) * 255
    return cv2.merge([mask, mask, mask])  # 3 canali (RGB)

# Interfaccia Gradio
iface = gr.Interface(
    fn=segment_hair,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Hair Segmentation",
    description="Carica un'immagine. Restituisce una maschera bianca per i capelli.",
    examples=["example.jpeg"]
)

if __name__ == "__main__":
    iface.launch()