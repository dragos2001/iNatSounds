import tarfile
from io import BytesIO
import os 
import tarfile
from multiprocessing import Pool, cpu_count
from .spectogram import partition_audio_stream, save_spectogram_image, generate_spectogram
import tqdm

out_dir = "/tmp/tmp_spectograms/train" 

from multiprocessing import Pool, cpu_count
from io import BytesIO

def process_audio_bytes(args):
    member_name, member_group, audio_bytes = args
    audio_stream = BytesIO(audio_bytes)
    chunks = partition_audio_stream(audio_stream)
    for i,chunk in enumerate(chunks):
        spectrogram = generate_spectogram(chunk)
        base_name = os.path.splitext(os.path.basename(member_name))[0]
        output_dir = os.path.join(out_dir,member_group)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_spectrogram_{i}")
        save_spectogram_image(spectrogram, output_path)
    return 1

def process_audio_from_tar(tar_path):
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar, Pool(cpu_count()) as pool:
        # generator yields (name, bytes) one by one
        results = pool.imap_unordered(
            process_audio_bytes,
            ((m.name,m.gname, tar.extractfile(m).read()) for m in stream_audio_members(tar))
        )
        for i, r in enumerate(results):
            count += r
            print(f"Completed {i} files")
    print(f"✅ Finished {count} audio files.")


def stream_audio_members(tar):
    for member in tar:
        if member.isfile() and member.name.lower().endswith((".wav", ".flac", ".mp3")):
            yield member

       


if __name__ == "__main__":
    tar_path = "/mnt/train/train.tar.gz"
    process_audio_from_tar(tar_path=tar_path)