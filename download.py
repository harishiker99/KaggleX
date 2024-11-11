from huggingface_hub import snapshot_download
model_id="harishnair04/Gemma-medtr-2b-sft-v2"
snapshot_download(repo_id=model_id, local_dir="Gemma-medtr-2b-sft-v2-hf",
                  local_dir_use_symlinks=False, revision="main")