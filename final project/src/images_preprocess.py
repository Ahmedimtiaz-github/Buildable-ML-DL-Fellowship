import os, sys, random
from glob import glob
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(ROOT, 'data', 'plant_disease')
OUT_DIR = os.path.join(ROOT, 'data', 'processed', 'plant_disease')
IMG_SIZE = (224,224)
os.makedirs(OUT_DIR, exist_ok=True)

classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
if not classes:
    # maybe images are directly inside INPUT_DIR or zipped folder with class dirs
    # find directories one level deeper
    subdirs = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for d in dirs:
            subdirs.append(os.path.join(root, d))
    classes = [os.path.relpath(d, INPUT_DIR) for d in subdirs] if subdirs else []
    
if not classes:
    print('No class subfolders detected in', INPUT_DIR); sys.exit(1)

print('Detected classes (first 20):', classes[:20])

# create train/val/test structure
for split in ['train','val','test']:
    for c in classes:
        os.makedirs(os.path.join(OUT_DIR, split, c), exist_ok=True)

# gather files per class and split
for c in classes:
    class_dir = os.path.join(INPUT_DIR, c)
    # if class_dir doesn't exist (maybe relpath case), try to find by matching folder name
    if not os.path.isdir(class_dir):
        # attempt to find folder anywhere matching the class name
        matches = [p for p in glob(os.path.join(INPUT_DIR, '**', c), recursive=True) if os.path.isdir(p)]
        if matches:
            class_dir = matches[0]
        else:
            print('Skipping class (not found):', c); continue
    files = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
        files.extend(glob(os.path.join(class_dir, '**', ext), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print('No images for class', c); continue
    # split
    train_files, temp_files = train_test_split(files, test_size=0.30, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    print(f'Class {c}: total {len(files)} -> train {len(train_files)}, val {len(val_files)}, test {len(test_files)}')

    # helper to save resized image
    def save_img(src_path, dst_path, augment=False):
        try:
            img = Image.open(src_path).convert('RGB')
            img = ImageOps.exif_transpose(img)
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            img.save(dst_path, quality=95)
            # optional augmentation: horizontal flip
            if augment:
                flipped = ImageOps.mirror(img)
                base,ext = os.path.splitext(dst_path)
                flipped.save(base + '_flip' + ext, quality=95)
        except Exception as e:
            print('Failed to process', src_path, e)

    # save train (with simple augmentation for small classes)
    augment = len(files) < 500  # heuristic: augment small classes
    for i,src in enumerate(train_files):
        dst = os.path.join(OUT_DIR, 'train', c, os.path.basename(src))
        save_img(src, dst, augment=augment)
    for src in val_files:
        dst = os.path.join(OUT_DIR, 'val', c, os.path.basename(src))
        save_img(src, dst, augment=False)
    for src in test_files:
        dst = os.path.join(OUT_DIR, 'test', c, os.path.basename(src))
        save_img(src, dst, augment=False)

print('Saved processed images to', OUT_DIR)
