import requests
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm  # for a nice progress bar
# Base S3 URL
SOURCE_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024"

# Files to transfer
FILES = [
    "train.tar.gz",
    "train.json.tar.gz",
    "val.tar.gz",
    "val.json.tar.gz",
    "test.tar.gz",
    "test.json.tar.gz"
]

# Azure account URL
ACCOUNT_URL = "https://mlworkspace1925296886.blob.core.windows.net/"

# SAS tokens per container
SAS_TOKENS = {
    "train": "?sp=racw&st=2025-11-24T23:53:11Z&se=2025-11-25T08:08:11Z&spr=https&sv=2024-11-04&sr=c&sig=Wr6VV2lg33W8LfR0%2BVTna6DWfZwL%2FMx721VIAI9qSWk%3D",
    "val": "?sp=racw&st=2025-11-25T08:16:15Z&se=2025-11-25T16:31:15Z&spr=https&sv=2024-11-04&sr=c&sig=bNR%2F6TuFnBEdzqO%2BHzyRDT8vLXUqwXrEcjLxxzPMJyA%3D",
    "test": "?sp=racw&st=2025-11-25T13:43:26Z&se=2025-11-25T21:58:26Z&spr=https&sv=2024-11-04&sr=c&sig=iZLoi1YrJqJk8ysQTBdmJHqZFKui9adOkAbgumpdRBo%3D"
}


for f in FILES:
    container = str(f.split(".")[0])
    print("Container:",container)
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=SAS_TOKENS[container]
    )

    container_client = blob_service_client.get_container_client(container)
    blob_client = container_client.get_blob_client(f)

    s3_url = f"{SOURCE_URL}/{f}"
    resp = requests.get(s3_url, stream=True)
    resp.raise_for_status()

    total_size = int(resp.headers.get("Content-Length", 0))
    progress = tqdm(total=total_size, unit="B", unit_scale=True, desc=f)

    def gen_chunks():
        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                progress.update(len(chunk))
                yield chunk

    # Let SDK handle block staging internally
    blob_client.upload_blob(gen_chunks(), overwrite=True, max_concurrency=4)

    progress.close()
    print(f"Finished uploading {f}")
