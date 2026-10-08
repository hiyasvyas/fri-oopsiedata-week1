from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="OopsieData-Submissions/WebVisualizer",
    repo_type="dataset",
    local_dir="data",
)
