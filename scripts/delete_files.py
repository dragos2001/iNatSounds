import os
import shutil
from iNatSounds.scripts.select_dataset_groups import ( get_relevant_measurements, select_relevant_groups, )

def delete_files(dir_path, relevant_groups):
    for file in os.listdir(dir_path):
        if file in relevant_groups:
            file_path = os.path.join(dir_path, file)
            shutil.rmtree(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"File not found, cannot delete: {file_path}")

if __name__ == "__main__":
    dirs_list2 = ["/mnt/tmp_wav/train", "/mnt/tmp_spectograms/train"]
    groups_dict = get_relevant_measurements(dirs_list2)
    relevant_groups = select_relevant_groups(groups_dict,down=False)
    print(f"Relevant groups: {relevant_groups}")
    delete_files("/mnt/tmp_wav/train", relevant_groups)
    