# I generated this code by an LLM, it should download a file from the link "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz" in the main folder and download the model meta-llama/Llama-2-7b-chat-hf and it's tokenizer in the subfolder "llama2-7b-local"
"""
download_assets.py

- Downloads ConceptNet assertions CSV.GZ to the current directory.
- Downloads meta-llama/Llama-2-7b-chat-hf (weights + tokenizer) to ./llama2-7b-local
- Optionally downloads any extra URLs you pass.

Usage examples:
  python download_assets.py
  HF_TOKEN=hf_xxx python download_assets.py
  python download_assets.py --hf-token hf_xxx --extra https://example.com/file1.zip --extra https://example.com/file2.txt
"""

import os
import sys
import argparse
import pathlib
import shutil
import time
from typing import Optional, List

# -------- streaming download with progress --------
def download_file(url: str, dest_path: pathlib.Path, chunk_size: int = 1024 * 1024) -> None:
    import requests
    from tqdm import tqdm

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists with non-zero size, skip
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[skip] {dest_path} already exists ({dest_path.stat().st_size} bytes).")
        return

    print(f"[download] {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"[ok] Saved to {dest_path.resolve()}")

# -------- HF model snapshot download --------
def download_llama2_local(
    repo_id: str = "meta-llama/Llama-2-7b-chat-hf",
    local_dir: str = "llama2-7b-local",
    hf_token: Optional[str] = None,
) -> None:
    """
    Uses huggingface_hub.snapshot_download to materialize the repo into local_dir.
    This fetches both model weights and tokenizer files.
    """
    from huggingface_hub import snapshot_download, login, HfHubHTTPError

    # Token resolution: CLI arg > env var > None
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            print(f"[warn] Could not login with provided token: {e}")

    target_dir = pathlib.Path(local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # We keep an on-disk cache inside local_dir/.hf-cache to avoid filling global cache
    cache_dir = target_dir / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download everything (weights + tokenizer + config).
    # For very constrained disks, you can restrict with allow_patterns.
    print(f"[hf] Downloading {repo_id} into {target_dir.resolve()} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # copy actual files into local_dir
            resume_download=True,
        )
    except HfHubHTTPError as e:
        print("\n[error] Hugging Face denied access.")
        print(" - Make sure you have requested & accepted access to Llama-2 on Hugging Face.")
        print(" - Provide a valid token via --hf-token or HF_TOKEN/HUGGINGFACE_HUB_TOKEN env var.")
        print(f" - Details: {e}\n")
        sys.exit(1)

    print(f"[ok] Llama-2 repo materialized at {target_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Download ConceptNet + Llama-2 assets.")
    parser.add_argument(
        "--conceptnet-url",
        default="https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
        help="URL for ConceptNet assertions .csv.gz",
    )
    parser.add_argument(
        "--conceptnet-out",
        default="conceptnet-assertions-5.7.0.csv.gz",
        help="Output filename in current directory",
    )
    parser.add_argument(
        "--local-llama-dir",
        default="llama2-7b-local",
        help="Folder to place the local Llama-2 model/tokenizer",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var)",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra URL(s) to download to the current directory (can repeat)",
    )
    args = parser.parse_args()

    
    conceptnet_dest = pathlib.Path(os.getcwd()) / args.conceptnet_out
    try:
        download_file(args.conceptnet_url, conceptnet_dest)
    except Exception as e:
        print(f"[error] Failed to download ConceptNet file: {e}")
        sys.exit(1)

    
    try:
        download_llama2_local(
            repo_id="meta-llama/Llama-2-7b-chat-hf",
            local_dir=args.local_llama_dir,
            hf_token=args.hf_token,
        )
    except Exception as e:
        print(f"[error] Failed to download Llama-2 repo: {e}")
        sys.exit(1)

    # 3) Any extra URLs (if you “also want to download …” more things)
    for url in args.extra:
        try:
            fname = url.strip("/").split("/")[-1] or f"download_{int(time.time())}"
            dest = pathlib.Path(os.getcwd()) / fname
            download_file(url, dest)
        except Exception as e:
            print(f"[warn] Skipped extra URL {url}: {e}")

    print("\nAll done ✅")

if __name__ == "__main__":
    main()
