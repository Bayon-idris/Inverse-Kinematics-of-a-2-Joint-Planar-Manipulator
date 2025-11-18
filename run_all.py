import subprocess
import sys

python = sys.executable  # Python du venv actif

scripts = ["data_gen.py", "train_anfis.py", "evaluate.py"]

for i, script in enumerate(scripts, 1):
    print(f"{i}) Running {script} ...")
    result = subprocess.run([python, script], check=True)
    if result.returncode != 0:
        print(f"Error running {script}, exit code {result.returncode}")
        break

print("Done. Check models/ and figs/ directories.")
