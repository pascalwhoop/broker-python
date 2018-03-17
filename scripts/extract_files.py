origin_dir = "/home/pascalwhoop/tank/Technology/Thesis/past_games/"
target_dir = "/home/pascalwhoop/tank/Technology/Thesis/past_games/extracted"

import tarfile
import os


def unzip_file(origin, dest):
    mode = "r:gz" if (origin.endswith("tar.gz")) else "r:"
    dir_name = origin.split("/")[-1].split(".")[0]
    target = os.path.join(dest, dir_name)
    tar = tarfile.open(origin, mode)
    tar.extractall(target)
    print("extracted {}".format(target.split("/")[-1]))


files = os.listdir(origin_dir)
os.makedirs(target_dir, exist_ok=True)

print("Extracting files")
files.sort()
for f in files:
    if "tar.gz" in f:
        unzip_file(os.path.join(origin_dir, f), target_dir)
