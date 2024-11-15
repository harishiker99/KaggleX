import subprocess

# Define the command
command = [
    "python", "llama.cpp/convert_hf_to_gguf.py", 
    "Gemma-medtr-2b-sft-v2-hf", 
    "--outfile", "Gemma-medtr-2b-sft-v2.gguf", 
    "--outtype", "q8_0"
]

# Run the command
subprocess.run(command, check=True)