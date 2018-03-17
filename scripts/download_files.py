import os
import csv
from urllib.request import urlretrieve
from progressbar import ProgressBar, Percentage, Bar

target_dir = "/home/pascalwhoop/tank/Technology/Thesis/past_games/"


def main():
    with open("scripts/finals_2017_06.games.csv") as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in list(csv_reader)[66:]:
            url = row[9]
            name = row[1]
            if "http" in url:
                download_file(url, name)


def download_file(url, game_name):
    os.makedirs(target_dir, exist_ok=True)
    file_name = os.path.join(target_dir, game_name) + ".tar.gz"
    print("Downloading file for game {}".format(game_name))
    local_filename = urlretrieve(url, file_name, dl_progress)
    print("Download complete. Target file: {}".format(local_filename))


# little logging helper for progress of file dls
p_bar = ProgressBar(widgets=[Percentage(), Bar()])


def dl_progress(count, blockSize, totalSize):
    count_blocks = totalSize / blockSize
    perc_steps = round(count_blocks / 100)
    if (count % perc_steps == 0):
        print("Downloading ... {}/{}".format(round(count / perc_steps), 100))


main()
