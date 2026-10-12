import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from mesh_alignment import process_and_save_alignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
PATH = os.getcwd()
print(f"Current working directory: {PATH}")

# Please inference the SAM 3D Body (3DB) Repo (https://github.com/facebookresearch/sam-3d-body) to get the 3DB Results
image_path = f"{PATH}/images/human_object/image.png"
mask_path = f"{PATH}/meshes/human_object/3DB_results/mask_human.png"
mesh_path = f"/path/to/ckpts/person_0.glb"
focal_length_json_path = f"{PATH}/meshes/human_object/3DB_results/focal_length.json"
output_dir = f"{PATH}/meshes/human_object/aligned_meshes"
os.makedirs(output_dir, exist_ok=True)

input_image = Image.open(image_path)
mask = Image.open(mask_path).convert('L')

success, output_mesh_path, result = process_and_save_alignment(
    mesh_path=mesh_path,
    mask_path=mask_path,
    image_path=image_path,
    output_dir=output_dir,
    device=device,
    focal_length_json_path=focal_length_json_path
)