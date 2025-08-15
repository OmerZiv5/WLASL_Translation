import cv2
import os
import shutil
import re
import random
import numpy as np
import matplotlib.pyplot as plt

# ==== CONFIGURATION ====
# make sure the path here matches to where you saved the raw videos and to where you want to save the extracted frames
root_input_folder = r"D:\Isolated WLASL Files\Test Videos" # this folder should contain your videos (train videos / test videos / combined train+test videos)
output_folder = r"D:\WLASL Files\Test Frames" # this folder will save the frames extracted from the videos you gave as input
differences_folder = r"D:\Isolated WLASL Files\Test Diff" # this folder will save the difference between the frames of the prev folder
frames_to_extract = 5
video_extensions = (".mp4", ".avi", ".mov", ".mkv")
num_repeats = 25
frame_offset_range = 5  # ± this many frames
saturation_scale = 1.5

os.makedirs(output_folder, exist_ok=True)


# ==== MAIN LOOP ====
def preprocess():
    category_sample_ids = {}  # counter per category

    for dirpath, _, filenames in os.walk(root_input_folder):
        for filename in filenames:
            if not filename.lower().endswith(video_extensions):
                continue

            video_path = os.path.join(dirpath, filename)

            # Category = first folder under root_input_folder
            rel_path = os.path.relpath(video_path, root_input_folder)
            category = rel_path.split(os.sep)[0]

            # Prepare category output folder
            category_output_dir = os.path.join(output_folder, category)
            os.makedirs(category_output_dir, exist_ok=True)

            # Initialize counter if new category
            if category not in category_sample_ids:
                category_sample_ids[category] = 1

            print(f"Processing video: {filename} (Category: {category})")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f" Cannot open {filename}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= frames_to_extract:
                print(f" Not enough frames in {filename}")
                cap.release()
                continue

            interval = total_frames // (frames_to_extract + 1)

            for repeat in range(num_repeats):
                # === Normal version sample folder ===
                sample_id = category_sample_ids[category]
                sample_folder = os.path.join(category_output_dir, f"{category}_{sample_id:06d}")
                os.makedirs(sample_folder, exist_ok=True)
                category_sample_ids[category] += 1

                for i in range(1, frames_to_extract + 1):
                    offset = random.randint(-frame_offset_range, frame_offset_range)
                    frame_index = i * interval + offset
                    frame_index = max(0, min(frame_index, total_frames - 1))

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"⚠ Could not read frame {frame_index} in {filename}")
                        continue

                    processed_frame = frame #boost_saturation_and_grayscale(frame)
                    frame_name = f"frame_{i}.jpg"
                    cv2.imwrite(os.path.join(sample_folder, frame_name), processed_frame)



            cap.release()

    print("Done generating samples!")


def clean_and_rename_categories(root_dir, min_files=5):
    """
    Deletes subfolders with fewer than min_files files,
    then renumbers remaining ones so IDs are contiguous.

    Assumes subfolder names are like 'category_00010'.
    """

    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Regex to match "category_id" with leading zeros allowed
        pattern = re.compile(rf"^{re.escape(category)}_(\d+)$")

        # Collect subfolders and their IDs
        subfolders = []
        for sub in os.listdir(category_path):
            match = pattern.match(sub)
            if match:
                sub_id = int(match.group(1))
                sub_path = os.path.join(category_path, sub)

                file_count = sum(
                    1 for f in os.listdir(sub_path)
                    if os.path.isfile(os.path.join(sub_path, f))
                )
                subfolders.append((sub_id, sub_path, file_count))

        # Sort by ID
        subfolders.sort(key=lambda x: x[0])

        # Step 1: Delete folders with too few files
        deleted_ids = set()
        for sub_id, sub_path, file_count in subfolders:
            if file_count < min_files:
                shutil.rmtree(sub_path)
                deleted_ids.add(sub_id)

        # Step 2: Renumber remaining ones
        shift_map = {}
        deleted_count_before = 0
        for sub_id, sub_path, file_count in subfolders:
            if sub_id in deleted_ids:
                deleted_count_before += 1
            else:
                new_id = sub_id - deleted_count_before
                shift_map[sub_id] = new_id

        # To avoid collisions, rename to a temp name first
        for sub_id, sub_path, file_count in sorted(
            ((i, p, c) for i, p, c in subfolders if i not in deleted_ids),
            key=lambda x: -x[0]  # rename highest IDs first
        ):
            new_id = shift_map[sub_id]
            new_name = f"{category}_{new_id:05d}"  # keep 5-digit padding
            temp_name = f"TEMP{new_id:05d}"
            temp_path = os.path.join(category_path, temp_name)
            os.rename(sub_path, temp_path)
            os.rename(temp_path, os.path.join(category_path, new_name))

def generate_frame_differences(input_root, output_root):
    """
    Goes over each sample folder under input_root, compares consecutive frames,
    and saves the absolute differences as images in output_root, preserving hierarchy.
    """
    for category in os.listdir(input_root):
        category_path = os.path.join(input_root, category)
        if not os.path.isdir(category_path):
            continue

        for sample_folder in os.listdir(category_path):
            sample_path = os.path.join(category_path, sample_folder)
            if not os.path.isdir(sample_path):
                continue

            # Read frame files in sorted order
            frame_files = sorted(
                [f for f in os.listdir(sample_path) if f.lower().endswith(".jpg")]
            )
            if len(frame_files) < 2:
                continue  # Need at least 2 frames to compare

            # Prepare output directory for this sample
            out_sample_path = os.path.join(output_root, category, sample_folder)
            os.makedirs(out_sample_path, exist_ok=True)

            for i in range(len(frame_files) - 1):
                frame1_path = os.path.join(sample_path, frame_files[i])
                frame2_path = os.path.join(sample_path, frame_files[i + 1])

                img1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

                if img1 is None or img2 is None:
                    continue

                # Compute absolute difference
                diff = cv2.absdiff(img1, img2)

                # Optional: enhance contrast for visibility
                diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

                diff_name = f"diff_{i+1}.jpg"
                cv2.imwrite(os.path.join(out_sample_path, diff_name), diff_enhanced)

    print(f"Frame differences generated in: {output_root}")


preprocess()
# in the next command - insert the path to the file that contains your original frames
generate_frame_differences(input_root=output_folder, output_root=differences_folder)
