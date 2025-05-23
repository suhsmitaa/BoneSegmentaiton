# Bone Segmentation and Landmark Detection

This repository contains a Python implementation for processing 3D CT images of the knee region to segment bones, expand contours, randomize masks, and detect anatomical landmarks on the tibia.

# Project Structure

- `main.py`: Main script implementing all tasks (segmentation, expansion, randomization, landmark detection).
- `results/`: Directory containing output masks (\*.nii.gz) and landmark coordinates (landmarks.txt).
- `requirements.txt`: List of available library

# working mechanism

- `Main Workflow`
  The main function orchestrates all tasks:
  Loads the CT scan.
  Segments bones and saves the original mask.
  Expands the mask by 2mm and 4mm, saving both.
  Generates two randomized masks and saves them.
  Detects tibial landmarks for all five masks and saves coordinates to landmarks.txt.
  All masks are saved in .nii.gz format using SimpleITK, and landmarks are written to a text file using standard Python file operations.

- `Key Parameters`
  VOXEL_SIZE_MM: Voxel size in mm (default: 1.0). Adjust based on ct_image.GetSpacing().
  MM_EXPANSION: Expansion distance for Task 1.2 (default: 2.0mm).
  RANDOM_SEED: Seed for randomization (default: 42) to ensure reproducible randomized masks.
  Outputs

- `Masks result`:

original_mask.nii.gz: Segmented femur and tibia.
expanded_mask_2mm.nii.gz: 2mm expanded mask.
expanded_mask_4mm.nii.gz: 4mm expanded mask.
randomized_mask_1.nii.gz: First randomized mask.
randomized_mask_2.nii.gz: Second randomized mask.
Landmarks:
landmarks.txt: Coordinates of medial and lateral lowest points for all five masks.
