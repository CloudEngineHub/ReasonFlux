import yaml
from tqdm import tqdm


def main() -> None:
    subset = ["extended", "diamond", "main"]
    setting = "generative_n_shot"
    for task in tqdm(subset):
        file_name = f"gpqa_{task}_{setting}.yaml"
        try:
            with open(f"{file_name}", "w") as f:
                f.write("# Generated by _generate_configs.py\n")
                yaml.dump(
                    {
                        "include": f"_gpqa_{setting}_yaml",
                        "task": f"gpqa_{task}_{setting}",
                        "dataset_name": f"gpqa_{task}",
                    },
                    f,
                )
        except FileExistsError:
            pass


if __name__ == "__main__":
    main()
