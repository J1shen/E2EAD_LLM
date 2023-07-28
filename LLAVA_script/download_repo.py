from huggingface_hub import snapshot_download
snapshot_download(repo_id="liuhaotian/llava-llama-2-13b-chat-lightning-preview",
                  cache_dir='./checkpoints')
snapshot_download(repo_id="liuhaotian/LLaVA-Instruct-150K",
                  cache_dir='./checkpoints',
                  repo_type='dataset')