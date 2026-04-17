import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Downloading Python code dataset (flytech/python-codes-25k)...")
    # This dataset contains instruction, input, and output (python code)
    dataset = load_dataset("flytech/python-codes-25k", split="train")
    
    # We will pick 5000 examples to keep the dataset size small for a Tiny-GPT
    # ~ 2-5 MB of raw text, perfect for training on a local machine to test
    subset = dataset.select(range(5000))
    
    out_file = os.path.join(os.path.dirname(__file__), "input.txt")
    print(f"Writing data to {out_file}...")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for item in tqdm(subset):
            instruction = item.get("instruction", "")
            code = item.get("output", "")
            
            # Format: We add special separators so the model learns
            # Question answering / code generation formats occasionally
            f.write(f"### INSTRUCTION:\n{instruction}\n\n")
            f.write(f"### CODE:\n{code}\n")
            f.write("-" * 50 + "\n\n")
            
    # Print some stats
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\nDone! Created {out_file} ({size_mb:.2f} MB).")
    
if __name__ == "__main__":
    main()
