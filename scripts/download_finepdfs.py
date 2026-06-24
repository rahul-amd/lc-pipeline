"""Download a small working set of FinePDFs (English) parquet chunks.

Grabs data/<config>/train000_{00000..N}.parquet from HuggingFaceFW/finepdfs into
a local folder. The dataset is public/ungated, so no HF login is required.

    python download_finepdfs.py                      # 5 chunks -> finpdf_sample/
    python download_finepdfs.py --num 5 --output_dir finpdf_sample
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main(argv=None):
    p = argparse.ArgumentParser("download_finepdfs")
    p.add_argument("--repo", default="HuggingFaceFW/finepdfs")
    p.add_argument("--config", default="eng_Latn")
    p.add_argument("--split", default="train")
    p.add_argument("--shard", type=int, default=0, help="shard group prefix (the 000 in 000_00000)")
    p.add_argument("--start", type=int, default=0, help="first chunk index")
    p.add_argument("--num", type=int, default=5, help="how many consecutive chunks")
    p.add_argument("--output_dir", default="finpdf_sample")
    args = p.parse_args(argv)

    patterns = [
        f"data/{args.config}/{args.split}/{args.shard:03d}_{i:05d}.parquet"
        for i in range(args.start, args.start + args.num)
    ]
    print(f"Downloading {len(patterns)} files from {args.repo} -> {args.output_dir}/")
    for pat in patterns:
        print("  ", pat)

    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=args.output_dir,
    )

    files = []
    for root, _, names in os.walk(args.output_dir):
        for n in names:
            if n.endswith(".parquet"):
                files.append(os.path.join(root, n))
    total = sum(os.path.getsize(f) for f in files)
    print(f"\nDone. {len(files)} parquet files, {total / 1e9:.2f} GB under {path}")
    for f in sorted(files):
        print(f"  {os.path.getsize(f) / 1e6:8.1f} MB  {f}")


if __name__ == "__main__":
    main()
