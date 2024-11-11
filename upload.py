from huggingface_hub import HfApi

# Replace 'your_api_token' with your actual Hugging Face API token
api_token = "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"
api = HfApi(token=api_token)

model_id = "harishnair04/Gemma-medtr-2b-sft-v2-gguf"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj="Gemma-medtr-2b-sft-v2.gguf",
    path_in_repo="Gemma-medtr-2b-sft-v2.gguf",
    repo_id=model_id,
)