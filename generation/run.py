import subprocess
import os

signatures_dir = "../images/signatures"
results_dir = "./results"

png_files = [f for f in os.listdir(signatures_dir) if f.endswith(".png")]


# Generator and Discriminator models (set dynamically)
gen_model = ["g_multipixelattention", "g_multivanilla"]
dis_model = ["d_snpixelattention", "d_vanilla"]

model_to_load = "results/2025-02-11_22-52-06/g_multipixelattention.pth"
amps_to_load = "results/2025-02-11_22-52-06/amps.pth"

# Iterate over each signature and execute the command
for signature in png_files:
    command = f"python main.py --root ../images/signatures/{signature} --dir-name {signature.split('.')[0]} --gen-model {gen_model[0]} --dis-model {dis_model[0]} \
        --seed 42 --num-steps 4000 > termOutput_{signature.split('.')[0]}.log 2>&1"
    
    # command = f"""
    #     python main.py --root ../images/signatures/{signature} --evaluation \
    #     --dir-name {signature.split('.')[0]} \
    #     --model-to-load {model_to_load} \
    #     --gen-model {gen_model[0]} --dis-model {dis_model[0]} \
    #     --amps-to-load {amps_to_load} \
    #     --num-steps 5 --batch-size 8 > termOutput_{signature.split('.')[0]}.log 2>&1
    #     """
    
    print(f"Executing: {command}")
    subprocess.run(command, shell=True, check=True)
