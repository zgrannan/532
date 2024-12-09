import json

from helpers import (
    get_output_model_name,
    get_pipeline_config_name,
    get_results_file_path,
)


def main():
    pass


if __name__ == "__main__":
    with open("../eval_configs/config.json", "r") as f:
        options = json.load(f)
    matrix = options["matrix"]

    runs = []

    for llm in matrix["pipeline_config.llm"]:
        for source in matrix["pipeline_config.sources"]:
            for r in matrix["finetune_config.r"]:
                runs.append((llm, source, r))

    for llm, source, r in runs:
        pipeline_config_name = get_pipeline_config_name(source, llm["model"])
        finetuned_model_name = get_output_model_name(pipeline_config_name, r)
        results_file_path = get_results_file_path(finetuned_model_name)
        results = json.load(open(results_file_path, "r"))
        llm_name = llm["model"] if "/" not in llm["model"] else llm["model"].split("/")[1]
        source_included = "yes" if source else "no"
        finetuned_wins = results["finetuned_wins"]
        base_wins = results["base_wins"]
        percentage = f"{finetuned_wins / (finetuned_wins + base_wins):.1%}"
        columns = [
            llm_name,
            source_included,
            str(r),
            str(finetuned_wins),
            str(base_wins),
            percentage.replace("%", "\\%"),
        ]
        print(" & ".join(columns) + " \\\\")

    main()
