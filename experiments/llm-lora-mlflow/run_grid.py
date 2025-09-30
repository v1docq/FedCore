import subprocess

MODELS = [
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-4B-Base",
    "mistralai/Mistral-7B-v0.1",
]
DATASETS = [
    ("qa", "squad"),
    ("summarization", "samsum"),
]

if __name__ == "__main__":
    for m in MODELS:
        for task, ds in DATASETS:
            run_name = f"{m.split('/')[-1]}_{ds}"
            cmd = [
                "python", "train_lora.py",
                "--model_id", m,
                "--task", task,
                "--dataset", ds,
                "--experiment", "LLM-LoRA",
                "--run_name", run_name,
            ]
            print("\n>>> RUN:", " ".join(cmd))
            subprocess.run(cmd, check=True)