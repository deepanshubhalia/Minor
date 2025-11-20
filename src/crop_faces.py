import os
from facenet_pytorch import MTCNN
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# FIX #1: Changed image_size from 224 to 160 to match the autoencoder input size
mtcnn = MTCNN(image_size=160, margin=10, post_process=True, device=device)

def crop_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for img_name in os.listdir(input_folder):
        try:
            img_path = os.path.join(input_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            face = mtcnn(img)
            if face is not None:
                save_path = os.path.join(output_folder, img_name)
                face_img = face.permute(1,2,0).mul(255).byte().cpu().numpy()
                Image.fromarray(face_img).save(save_path)
                print(f"✓ Processed: {img_name}")
        except Exception as e:
            print("Error:", img_name, e)
    print("Done cropping:", input_folder)

if __name__ == "__main__":
    crop_folder("data/faces_real", "data/cropped_real")
    crop_folder("data/faces_fake", "data/cropped_fake")
