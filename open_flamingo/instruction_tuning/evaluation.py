# import re
# import json
# import argparse
# from tqdm import tqdm


# def extract_answer_regex(text: str) -> str:
#     """
#     Extracts a hyphenated word following 'Final Answer: ' using regex.

#     Args:
#         text: The input string to search.

#     Returns:
#         The extracted word (e.g., 'Mild-Dementia') or None if no match is found.
#     """
#     # This pattern looks for "Final Answer:", followed by optional whitespace,
#     # and then captures a sequence of word characters and hyphens.
#     match = re.search(r"Final Answer:\s*([\w-]+)", text)

#     if match:
#         return match.group(1)  # Return the first captured group
#     return "None"


# if __name__ == "__main__":

#     default_inference_result = "/mnt/data/nict/maund/baseline_models/instruct_flamingo/predictions_validation/0821-clever_flamingo_v2_3b-2k_context-40G-resume-from-mpt7b-checkpoint-03-checkpoint_15/eval_dataset_config_-1/llava-mri-cot-1k-test_all.json"

#     # make inference_result path a command line argument: --inference_result

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--inference_result", type=str, default=default_inference_result
#     )
#     args = parser.parse_args()

#     inference_result = args.inference_result

#     with open(inference_result, "r") as f:
#         data = json.load(f)

#     correct_format = 0
#     match = 0

#     for result in tqdm(data):
#         output = result["output"]
#         target = result["target"]

#         answer = extract_answer_regex(output)
#         if answer and target:
#             answer = answer.lower().replace(" ", "")
#             target = target.lower().replace(" ", "")

#             print("---------")
#             if "-" in answer:
#                 correct_format += 1
#             if (not (answer == target)) and ("-" not in answer):
#                 print(f"Output: {output}")
#                 print(f"Answer: {answer}")
#                 print(f"Target: {target}")
#             if answer == target:
#                 match += 1

#     # Get inference folder from inference_result path
#     inference_folder = "/".join(inference_result.split("/")[:-1])

#     accuracy = match / len(data) * 100
#     correct_format = correct_format / len(data) * 100

#     result = {
#         "accuracy": accuracy,
#         "correct_format": correct_format,
#         "num_samples": len(data),
#     }

#     with open(f"{inference_folder}/evaluation.json", "w") as f:
#         json.dump(result, f, indent=4)

#     print(f"Accuracy: {accuracy}%")
#     print(f"Correct Format: {correct_format}%")


import re
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score  # <-- ADDED: Import f1_score


def extract_answer_regex(text: str) -> str:
    """
    Extracts a hyphenated word following 'Final Answer: ' using regex.
    """
    match = re.search(r"Final Answer:\s*([\w-]+)", text)
    if match:
        return match.group(1)
    return "None"


if __name__ == "__main__":
    default_inference_result = "/mnt/data/nict/maund/baseline_models/instruct_flamingo/predictions_validation/0821-clever_flamingo_v2_3b-2k_context-40G-resume-from-mpt7b-checkpoint-03-checkpoint_15/eval_dataset_config_-1/llava-mri-cot-1k-test_all.json"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_result", type=str, default=default_inference_result
    )
    args = parser.parse_args()
    inference_result = args.inference_result

    with open(inference_result, "r") as f:
        data = json.load(f)

    correct_format = 0
    match = 0

    # --- ADDED: Lists to store all labels for F1 calculation ---
    y_true = []
    y_pred = []

    for result in tqdm(data):
        output = result["output"]
        target = result["target"]
        answer = extract_answer_regex(output)

        if answer and target:
            output = output.lower()
            answer = answer.lower().replace(" ", "")
            target = target.lower().replace(" ", "")

            # --- ADDED: Append labels to our lists ---
            y_true.append(target)
            if output == target:
                y_pred.append(output)
            else:
                y_pred.append(answer)

            print("---------")
            if "-" in answer or "-" in output:
                correct_format += 1
            if (not (answer == target)) and ("-" not in answer):
                print(f"Output: {output}")
                print(f"Answer: {answer}")
                print(f"Target: {target}")
            if answer == target or output == target:
                match += 1

    inference_folder = "/".join(inference_result.split("/")[:-1])

    accuracy = match / len(data) * 100
    correct_format = correct_format / len(data) * 100

    # --- ADDED: Calculate F1 scores ---
    # 'macro': Calculate metrics for each label, and find their unweighted mean.
    #          This does not take label imbalance into account.
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    # 'weighted': Calculate metrics for each label, and find their average
    #             weighted by support (the number of true instances for each label).
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100

    result = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,  # <-- ADDED
        "f1_weighted": f1_weighted,  # <-- ADDED
        "correct_format": correct_format,
        "num_samples": len(data),
    }

    # Save confusion matrix for further analysis
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"{inference_folder}/confusion_matrix.png")
    print(f"Confusion matrix saved to {inference_folder}/confusion_matrix.png")

    with open(f"{inference_folder}/evaluation.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score (Macro): {f1_macro:.2f}%")  # <-- ADDED
    print(f"F1 Score (Weighted): {f1_weighted:.2f}%")  # <-- ADDED
    print(f"Correct Format: {correct_format:.2f}%")
