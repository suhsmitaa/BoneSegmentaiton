import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, distance_transform_edt
import random


INPUT_CT_PATH = "image/3702_left_knee.nii.gz"  
OUTPUT_DIR = "results"
MM_EXPANSION = 2.0  
VOXEL_SIZE_MM = 1.0  
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_ct_image(file_path):
    image = sitk.ReadImage(file_path)
    return image

def segment_bones(ct_image, lower_threshold=200, upper_threshold=3000):
    ct_array = sitk.GetArrayFromImage(ct_image)
    bone_mask = (ct_array >= lower_threshold) & (ct_array <= upper_threshold)
    bone_mask_sitk = sitk.GetImageFromArray(bone_mask.astype(np.uint8))
    bone_mask_sitk.CopyInformation(ct_image)
    return bone_mask_sitk

def expand_mask(mask, expansion_mm, voxel_size_mm):
    mask_array = sitk.GetArrayFromImage(mask)
    iterations = int(expansion_mm / voxel_size_mm)
    expanded_mask = binary_dilation(mask_array, iterations=iterations)
    expanded_mask_sitk = sitk.GetImageFromArray(expanded_mask.astype(np.uint8))
    expanded_mask_sitk.CopyInformation(mask)
    return expanded_mask_sitk

def randomize_mask(original_mask, expanded_mask, expansion_mm, voxel_size_mm):
    original_array = sitk.GetArrayFromImage(original_mask)
    expanded_array = sitk.GetArrayFromImage(expanded_mask)
    distance_map = distance_transform_edt(original_array)
    max_distance = expansion_mm / voxel_size_mm
    random.seed(RANDOM_SEED)
    random_threshold = random.uniform(0, max_distance)
    randomized_mask = (distance_map <= random_threshold) & (distance_map > 0)
    randomized_mask = randomized_mask | original_array  
    randomized_mask_sitk = sitk.GetImageFromArray(randomized_mask.astype(np.uint8))
    randomized_mask_sitk.CopyInformation(original_mask)
    return randomized_mask_sitk

#Find medial and lateral lowest points on the tibial surface
def find_tibial_landmarks(mask):
    mask_array = sitk.GetArrayFromImage(mask)
    z, y, x = np.where(mask_array)
    lower_half_indices = z > (mask_array.shape[0] // 2)
    z, y, x = z[lower_half_indices], y[lower_half_indices], x[lower_half_indices]
    if len(z) == 0:
        return None, None  
    
    max_z = z.max()
    lowest_points = np.where(z == max_z)[0]
    y_lowest, x_lowest = y[lowest_points], x[lowest_points]
    medial_idx = np.argmin(x_lowest)  
    lateral_idx = np.argmax(x_lowest)  
    medial_point = (z[lowest_points[medial_idx]], y[lowest_points[medial_idx]], x_lowest[medial_idx])
    lateral_point = (z[lowest_points[lateral_idx]], y[lowest_points[lateral_idx]], x_lowest[lateral_idx])
    return medial_point, lateral_point

def save_mask(mask, output_path):
    sitk.WriteImage(mask, output_path)

def save_landmarks(landmarks, output_path):
    with open(output_path, 'a') as f:
        for mask_name, (medial, lateral) in landmarks.items():
            f.write(f"{mask_name}:\n")
            f.write(f"  Medial Point: {medial}\n")
            f.write(f"  Lateral Point: {lateral}\n\n")

def main():
    ct_image = load_ct_image(INPUT_CT_PATH)
    
    #Bone Segmentation
    bone_mask = segment_bones(ct_image)
    save_mask(bone_mask, os.path.join(OUTPUT_DIR, "original_mask.nii.gz"))
    
    #Contour Expansion (2mm)
    expanded_mask_2mm = expand_mask(bone_mask, MM_EXPANSION, VOXEL_SIZE_MM)
    save_mask(expanded_mask_2mm, os.path.join(OUTPUT_DIR, "expanded_mask_2mm.nii.gz"))
    
    #Contour Expansion (4mm)
    expanded_mask_4mm = expand_mask(bone_mask, 4.0, VOXEL_SIZE_MM)
    save_mask(expanded_mask_4mm, os.path.join(OUTPUT_DIR, "expanded_mask_4mm.nii.gz"))
    
    #Randomized Contour Adjustment
    random_mask_1 = randomize_mask(bone_mask, expanded_mask_2mm, MM_EXPANSION, VOXEL_SIZE_MM)
    save_mask(random_mask_1, os.path.join(OUTPUT_DIR, "randomized_mask_1.nii.gz"))
    
    # Generate second randomized mask with a different seed
    random.seed(RANDOM_SEED + 1)
    random_mask_2 = randomize_mask(bone_mask, expanded_mask_2mm, MM_EXPANSION, VOXEL_SIZE_MM)
    save_mask(random_mask_2, os.path.join(OUTPUT_DIR, "randomized_mask_2.nii.gz"))
    
    #Landmark Detection
    masks = {
        "Original Mask": bone_mask,
        "2mm Expanded Mask": expanded_mask_2mm,
        "4mm Expanded Mask": expanded_mask_4mm,
        "Randomized Mask 1": random_mask_1,
        "Randomized Mask 2": random_mask_2
    }
    landmarks = {}
    for mask_name, mask in masks.items():
        medial, lateral = find_tibial_landmarks(mask)
        landmarks[mask_name] = (medial, lateral)
    

    save_landmarks(landmarks, os.path.join(OUTPUT_DIR, "landmarks.txt"))

if __name__ == "__main__":
    main()