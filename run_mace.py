import subprocess

# Define your iterables
sizes = ['-1k', '-10k', '-100k', '']
models = ['slider', 'prepend', 'PKV']

def run_study():
    for size in sizes:
        for model in models:
            # Skip the specific edge case
            if model == 'PKV' and size == '-1k':
                print(f"Skipping: model={model}, size={size}")
                continue
            
            # Construct the file path
            postprocessed_path = f'_artifacts/dataset_size_study/mgen_ft-den{size}-{model}_post.parquet'
            
            print(f"Processing: {postprocessed_path}")
            
            # Construct the shell command
            command = [
                "python", "_utils/_metrics/mace_ehull.py",
                "--post_parquet", postprocessed_path,
                "--output_parquet", postprocessed_path,
                "--num_workers", "16"
            ]
            
            # Execute the command
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while processing {postprocessed_path}: {e}")

if __name__ == "__main__":
    run_study()