echo "Downloading dataset and annotations"
FILES=("train.tar.gz" "train.json.tar.gz" "validation.tar.gz" "validation.json.tar.gz" "test.tar.gz" "test.tar.json.gz")
BLOB_URLS=("https://mlworkspace1925296886.blob.core.windows.net/train"
"https://mlworkspace1925296886.blob.core.windows.net/validation"
"https://mlworkspace1925296886.blob.core.windows.net/test"
)
SAS_TOKENS=("?sp=racw&st=2025-11-24T23:53:11Z&se=2025-11-25T08:08:11Z&spr=https&sv=2024-11-04&sr=c&sig=Wr6VV2lg33W8LfR0%2BVTna6DWfZwL%2FMx721VIAI9qSWk%3D"
"?sp=r&st=2025-11-25T00:01:01Z&se=2025-11-25T08:16:01Z&spr=https&sv=2024-11-04&sr=c&sig=4l%2F7WgcaVasNf%2BiQcmlglgCsq446JcB%2FcyRKGhIBAzw%3D"
"?sp=racw&st=2025-11-24T23:52:21Z&se=2025-11-25T08:07:21Z&spr=https&sv=2024-11-04&sr=c&sig=EVfZQNxYkkHwExTCLVs0Wmgh2LFno0rma%2FZB5T1csi4%3D"
)

for i in "${!FILES[@]}"; do
  echo "Copying ${FILES[$i]}"

  if [[ "${FILES[$i]}" == train* ]]; then
    DEST="${BLOB_URLS[0]}/${FILES[$i]}${SAS_TOKENS[0]}"
  elif [[ "${FILES[$i]}" == validation* ]]; then
    DEST="${BLOB_URLS[1]}/${FILES[$i]}${SAS_TOKENS[1]}"
  elif [[ "${FILES[$i]}" == test* ]]; then
    DEST="${BLOB_URLS[2]}/${FILES[$i]}${SAS_TOKENS[2]}"
  fi
  azcopy copy "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024/${FILES[$i]}" \
  "$DEST"  \
  --overwrite=true \
  --from-to=S3Blob
done

