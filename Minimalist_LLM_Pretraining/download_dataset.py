import os
import argparse
from huggingface_hub import hf_hub_download


def get(fname, local_dir):
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="yorkerlin/c4-subset", filename=fname,
                        repo_type="dataset", local_dir=local_dir)


num_chunks = 200
local_dir = os.path.join(os.path.dirname(__file__), 'dataset/c4-t5/subset/')

parser = argparse.ArgumentParser(description="Download a subset of C4 shards.")
parser.add_argument(
    "--num-chunks",
    type=int,
    default=num_chunks,
    help="Number of training chunks to download (default: %(default)s)",
)
parser.add_argument(
    "--save-dir",
    type=str,
    default=local_dir,
    help="Directory to save downloaded files (default: script dataset path)",
)
args = parser.parse_args()

num_chunks = args.num_chunks
local_dir = args.save_dir
os.makedirs(local_dir, exist_ok=True)

for i in range(num_chunks):
    get("c4-train.%05d-of-01024.json.gz" % i, local_dir)
for i in range(8):
    get("c4-validation.%05d-of-00008.json.gz" % i, local_dir)
