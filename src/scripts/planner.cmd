@tail -n +3 '%~f0' | python - %*
@goto :eof

import dotenv, os, sys, tqdm, traceback
from cryptography.fernet import Fernet
from pathlib import Path
from tqdm import tqdm

dotenv.load_dotenv('.env')
cipher = Fernet(os.environ['planner.key'])
args = sys.argv[1:]

if args[0] == 'help':
    print('Use: planner (open|close|help)')
    exit()

def map_dir(src_dir, dst_dir, operation):
    try:
        for root, dirs, files in tqdm(os.walk(src_dir)):
            if root == 'planner\\survey\\pdf files': continue

            relative_path = Path(root).relative_to(src_dir)
            target_root = dst_dir / relative_path
            target_root.mkdir(parents=True, exist_ok=True)

            for file in files:
                src_file = Path(root) / file
                dst_file = target_root / file

                if operation=='encrypt':
                    with open(src_file, "r", encoding="utf-8") as f_in, \
                         open(dst_file, "wb") as f_out:
                        for line in f_in:
                            encrypted = cipher.encrypt(line.encode("utf-8"))
                            f_out.write(encrypted + b"\n")
                else:
                    with open(src_file, "rb") as f_in, \
                         open(dst_file, "w", encoding="utf-8") as f_out:
                        for line in f_in:
                            encrypted = line.strip()
                            if not encrypted: continue
                            decrypted = cipher.decrypt(encrypted)
                            f_out.write(decrypted.decode("utf-8"))
    except:
        print('root, dirs, files, file', root, dirs, files, file)
        traceback.print_exc()

if args[0] == 'open':
    map_dir('planner_vault', 'planner', 'decrypt')
    exit()

if args[0] == 'close':
    map_dir('planner', 'planner_vault', 'encrypt')
    exit()
