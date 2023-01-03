import argparse
import os
import random
import shutil
import tqdm

parser = argparse.ArgumentParser(description='create trainset')
parser.add_argument('--imagenet_path', type=str, default='/home/zeyuan.yin/imagenet/train', help='path to imagenet')
parser.add_argument('--dest_dir', type=str, default='/home/zeyuan.yin/imagenet/train_50', help='path to imagenet')
args = parser.parse_args()

imagenet_path = args.imagenet_path
dest_dir = args.dest_dir

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Iterate through the class directories in ImageNet
for class_dir in tqdm.tqdm(os.listdir(imagenet_path)):
    class_path = os.path.join(imagenet_path, class_dir)
    if not os.path.isdir(class_path):
        continue

    # Select 50 random images from the class directory
    images = random.sample(os.listdir(class_path), 50)

    # Create a destination directory for the class
    class_dest_dir = os.path.join(dest_dir, class_dir)
    if not os.path.exists(class_dest_dir):
        os.makedirs(class_dest_dir)

    # Copy the selected images to the destination directory
    for image in images:
        src = os.path.join(class_path, image)
        dest = os.path.join(class_dest_dir, image)
        shutil.copy(src, dest)

print('Done!')


#  check the number of images in subset-source/train and subset-source/val
#  $ find train_50/ -name "*.JPEG" | wc -l
#  50000

os.system(f'find {dest_dir} -name "*.JPEG" | wc -l')
