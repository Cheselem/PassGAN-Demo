import urllib.request
import os

url = "https://github.com/brannondorsey/PassGAN/releases/download/data/rockyou-train.txt"
output_file = "data/crypto_ai_dataset.txt"
target_size = 25 * 1024 * 1024  # 25 MB

print(f"Downloading dataset from {url}...")
req = urllib.request.urlopen(url)

bytes_written = 0
with open(output_file, "wb") as f:
    while bytes_written < target_size:
        # Read in chunks
        chunk = req.read(1024 * 1024)
        if not chunk:
            break
        
        # If writing this chunk exceeds target size, we truncate it at the last newline
        if bytes_written + len(chunk) > target_size:
            excess = (bytes_written + len(chunk)) - target_size
            cut_chunk = chunk[:-excess]
            # Ensure we end on a newline so we don't have a malformed password
            last_newline = cut_chunk.rfind(b'\n')
            if last_newline != -1:
                cut_chunk = cut_chunk[:last_newline + 1]
            f.write(cut_chunk)
            bytes_written += len(cut_chunk)
            break
        else:
            f.write(chunk)
            bytes_written += len(chunk)

print(f"Dataset saved to {output_file}. Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
