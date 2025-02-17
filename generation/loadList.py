import os

signatures_dir = "../images/signatures"
results_dir = "./results"

png_files = [f for f in os.listdir(signatures_dir) if f.endswith(".png")]

processed_results = []
for subdir in os.listdir(results_dir):
    subdir_path = os.path.join(results_dir, subdir)
    if os.path.isdir(subdir_path) and "$" in subdir:
        processed_results.append(subdir.split("$")[0] + ".png")

# Print results
print("PNG Files in Signatures:", png_files)
print("Processed Subdirectory Names:", processed_results)