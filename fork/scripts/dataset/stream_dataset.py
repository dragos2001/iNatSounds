"""
SPDX-License-Identifier: MIT
Copyright © 2026 dragos2001
"""
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient,BlobClient,ContainerClient
import tarfile
from io import BytesIO
import os 
import tqdm
from tqdm import tqdm
from multiprocessing import Pool, TimeoutError

def connect_to_clients(account_url: str, container_list: list) :
    """
    Connect to Azure Blob Storage using DefaultAzureCredential.

    Args:
        account_url (str): The URL of the Azure Blob Storage account.

    Returns:
        BlobServiceClient: An instance of BlobServiceClient connected to the specified account.
    """
    try:
                #connect through default azure credential
                credential = DefaultAzureCredential()

                #blob service client
                blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
                
                #container clients
                train_container_client = blob_service_client.get_container_client(container_list[0])
                validation_container_client = blob_service_client.get_container_client(container_list[1])
                test_container_client = blob_service_client.get_container_client(container_list[2])
    except Exception as e:
                print(f"An error occurred: {e}")
                raise e    

    print("Connected to Blob Storage containers successfully.")

    return  train_container_client,validation_container_client, test_container_client

def process_blob(container_client: ContainerClient, blob_name: str):
    """
    Extract a blob from Azure Blob Storage and save it to a local path.

    Args:
        container_client (ContainerClient): The container client to access the blob.
        blob_name (str): The name of the blob to extract.
        extract_path (str): The local path to save the extracted blob.
    """

    dataset_split_name= blob_name.split('.')[0]
    try:
        #get blob client
        blob_client = container_client.get_blob_client(blob_name)


        # get blob properties for size (for progress)
        try:
            props = blob_client.get_blob_properties()
            total_size = getattr(props, "size", None)
        except Exception:
            total_size = None




        print(f"Starting extraction of blob {blob_name}...")
        

        #in case json file
        if "json" in blob_name:
            #create json extract file apth
            os.makedirs("./dataset/annotations/", exist_ok=True)
            file_path = os.path.join("./dataset/annotations/", f"{dataset_split_name}.json")
            
            if os.path.exists(file_path):
                
                print(f"File {file_path} already exists. Skipping download.")
            
            else:
                #download blob to a stram
                download_stream = blob_client.download_blob()

                #write the downloaded stream to a filepath
                with open(file_path, mode="wb") as sample_blob:
                    with tqdm(range(0,100), total=total_size, desc=f"Download {blob_name}", disable=total_size is None) as pbar:
                        for chunk in download_stream.chunks():
                            sample_blob.write(chunk)
                            pbar.update(len(chunk))
                    
                #confirm extraction
                print(f"Blob {blob_name} extracted successfully to {file_path}.")

        #in case dataset
        else:
            #download blob to a stream
            download_stream = blob_client.download_blob()

            #create dataset extract path
            os.makedirs(f"./dataset/{dataset_split_name}/", exist_ok=True)
            data_stream = BytesIO()
            download_stream.readinto(data_stream)

            with tarfile.open(fileobj=data_stream, mode="r:gz") as tar_obj:
               tar_obj.list()
                
            print(f"Blob {blob_name} extracted  names extracted succesfully:")
            

            #print(f"Blob {blob_name} extracted  names successfully to {extract_path}.")
    except Exception as e:
        print(f"An error occurred while extracting blob {blob_name}: {e}")
        raise e


def process_container(container_client: ContainerClient, blob_names: list):
    """
    Process multiple blobs in a container and extract them to specified local paths.

    Args:
        container_client (ContainerClient): The container client to access the blobs.
        blob_names (list): A list of blob names to extract.
        extract_paths (list): A list of local paths to save the extracted blobs.
    """

    print("Blob names:", list(container_client.list_blob_names()))
    for blob_name in blob_names:
        process_blob(container_client, blob_name)

def main():

    #storage account url
    account_url = "https://mlworkspace1925296886.blob.core.windows.net/"

    #list of containers 
    container_list = ["train", "val", "test"]
    
    #tar files 
    dataset_archives = ["train.tar.gz", "val.tar.gz", "test.tar.gz"]
    annotations_file = ["train.json.tar.gz", "val.json.tar.gz", "test.json.tar.gz"]

    #train, val and test container clients
    train_client, val_client, test_client = connect_to_clients(account_url, container_list)

    process_parallel = False
    if process_parallel:
        with Pool(processes=3) as pool:
            #process train container
            pool.map(process_container, 
                    [(train_client, [annotations_file[0], dataset_archives[0]]),
                    (val_client, [annotations_file[1], dataset_archives[1]]),
                    (test_client, [annotations_file[2], dataset_archives[2]])]
                    )
    else:    
        #process test container
        process_container(test_client, [annotations_file[2], dataset_archives[2]])


if __name__ == "__main__":
    main()