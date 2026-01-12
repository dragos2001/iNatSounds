"""
SPDX-License-Identifier: MIT
Copyright © 2026 dragos2001
"""
import os 


def get_relevant_measurements(dirs_list):
    groups_dict={}
    for dataset_dir in dirs_list:
        for species_dir in os.listdir(dataset_dir):
            if groups_dict.get(species_dir) is None:
                groups_dict[species_dir] = 1
            else:
                groups_dict[species_dir] += 1
    return groups_dict

def select_relevant_groups(groups_dict,threshold=1,down=False):
    if down:
        return [k for k, v in groups_dict.items() if v <= threshold]
    return [k for k, v in groups_dict.items() if v > threshold]

if __name__ == "__main__":
    train_dir = "/mnt/train/train_spectograms"
    test_dir = "/mnt/test/test_spectograms/test"
    valid_dir = "/mnt/val/val_spectograms"
    dirs_list = [train_dir,test_dir,valid_dir]
    dirs_list2 = ["/mnt/tmp_wav/train", "/mnt/tmp_spectograms/train"]
    groups_dict = get_relevant_measurements(dirs_list2)
    relevant_groups = select_relevant_groups(groups_dict,down=False)
    print(f"Relevant groups: {relevant_groups}")
    with open("/mnt/relevant_groups.txt","w") as f:
        for group in relevant_groups:
            f.write(f"{group}\n")