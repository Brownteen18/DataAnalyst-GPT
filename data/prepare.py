import os
import tiktoken
import numpy as np

def main():
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    if not os.path.exists(input_file_path):
        print("input.txt not found! Please run curate_data.py first.")
        return
        
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    print(f"Data has {n} characters.")
    print(f"Train split has {len(train_data)} characters.")
    print(f"Val split has {len(val_data)} characters.")
    
    print("Encoding with GPT-2 tiktoken BPE...")
    enc = tiktoken.get_encoding("gpt2")
    
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")
    
    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    print("train.bin and val.bin created successfully.")

if __name__ == '__main__':
    main()
