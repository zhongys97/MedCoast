from huggingface_hub import snapshot_download
import os

# """
# ============ DOWNLOAD MedQA ============ 
# """
# download_url = "https://huggingface.co/datasets/bigbio/med_qa/resolve/main/data_clean.zip?download=true"
# os.system(f"wget {download_url} -O ./MedQA.zip")
# os.makedirs("./original/MedQA", exist_ok=True)
# os.system("unzip ./MedQA.zip -d ./original/MedQA")
# os.system("rm ./MedQA.zip")

# """
# ============ DOWNLOAD MedMCQA ============ 
# """
# dataset_id = "openlifescienceai/medmcqa"
# local_path = "./original/MedMCQA" 

# os.makedirs(local_path, exist_ok=True)

# downloaded_path = snapshot_download(
#     repo_id=dataset_id,
#     local_dir=local_path,
#     repo_type="dataset"
# )


# """
# ============ DOWNLOAD PubMedQA ============ 
# """
# dataset_id = "qiaojin/PubMedQA"
# local_path = "./original/PubMedQA" 

# os.makedirs(local_path, exist_ok=True)

# downloaded_path = snapshot_download(
#     repo_id=dataset_id,
#     local_dir=local_path,
#     repo_type="dataset"
# )


# """
# ============ DOWNLOAD MedXpertQA ============ 
# """
# dataset_id = "TsinghuaC3I/MedXpertQA"
# local_path = "./original/MedXpertQA" 

# os.makedirs(local_path, exist_ok=True)

# downloaded_path = snapshot_download(
#     repo_id=dataset_id,
#     local_dir=local_path,
#     repo_type="dataset"
# )


# """
# ============ DOWNLOAD MedQuAD ============ 
# """
# dataset_id = "lavita/MedQuAD"
# local_path = "./original/MedQuAD"

# os.makedirs(local_path, exist_ok=True)

# downloaded_path = snapshot_download(
#     repo_id=dataset_id,
#     local_dir=local_path,
#     repo_type="dataset"
# )


# """
# ============ DOWNLOAD MMLU Med ============ 
# """
# dataset_id = "cais/mmlu"
# local_path = "./original/MMLU"

# os.makedirs(local_path, exist_ok=True)

# downloaded_path = snapshot_download(
#     repo_id=dataset_id,
#     local_dir=local_path,
#     repo_type="dataset"
# )


# """
# ============ DOWNLOAD BioASQ ============ 
# """
dataset_id = "kroshan/BioASQ"
local_path = "./original/BioASQ" 

os.makedirs(local_path, exist_ok=True)

downloaded_path = snapshot_download(
    repo_id=dataset_id,
    local_dir=local_path,
    repo_type="dataset"
)

