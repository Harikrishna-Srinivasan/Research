import sys
from huggingface_hub import HfApi, list_repo_files

def resolve_gguf(repo_id: str):
    print(f"Resolving: {repo_id}")
    api = HfApi()
    
    # Check if original repo has gguf
    try:
        files = list_repo_files(repo_id)
        gguf_files = [f for f in files if f.endswith(".gguf")]
        if gguf_files:
            return f"Found in original repo: {repo_id}"
    except Exception as e:
        print(f"Original repo failed: {e}")

    # Not found, search for GGUF
    # For example if repo_id is deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # We want to find GGUF quants of this exact original model. 
    # Usually they have the model name in it, so we can search by the model basename
    model_basename = repo_id.split("/")[-1]
    print(f"Searching for GGUF of {model_basename}...")
    models = api.list_models(
        search=model_basename,
        tags="gguf",
        sort="downloads"
    )
    
    # We want to get the first one that has GGUF files in it
    # We also might want to check if the model repo name implies it's a quant of our target model
    models = list(models)
    if not models:
        return "No GGUF equivalents found."
        
    for m in models:
        # print(f"Candidate: {m.id} (downloads: {m.downloads})")
        # Try finding a suitable file
        try:
            files = list_repo_files(m.id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if gguf_files:
                q4 = [f for f in gguf_files if "q4_k_m" in f.lower()]
                best = q4[0] if q4 else gguf_files[0]
                return f"Best auto-resolution: {m.id} / {best}"
        except Exception:
            pass

    return "Failed to find any playable files."

print(resolve_gguf('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'))
print(resolve_gguf('google/gemma-3-4b-it'))
print(resolve_gguf('mistralai/Ministral-8B-Instruct-2412'))
