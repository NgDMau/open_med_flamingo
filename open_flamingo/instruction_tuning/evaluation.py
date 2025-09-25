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

    valid_classes = {"non-dementia", "mild-dementia", "moderate-dementia"}

    results_to_save = []

    for result in tqdm(data):
        output = result["output"]
        target = result["target"]
        answer = extract_answer_regex(output)
        
        final_answer = ""

        if output and target:
            output = output.lower()
            answer = answer.lower().replace(" ", "")
            target = target.lower().replace(" ", "")

            # --- ADDED: Append labels to our lists ---
            y_true.append(target)
            if output == target:
                y_pred.append(output)
                final_answer = output
            elif answer in valid_classes:
                y_pred.append(answer)
                final_answer = answer
            elif output in valid_classes:
                y_pred.append(output)
                final_answer = output
            else:
                last_class = ""
                for class_name in valid_classes:
                    if class_name in output:
                        y_pred.append(class_name)
                        last_class = class_name
                        final_answer = class_name
                        break
                if last_class == "":
                    y_pred.append("invalid")  # For any invalid prediction
                    final_answer = answer
                
                print(f"Output: {output}")
                print(f"Answer: {answer}")
                print(f"Target: {target}")

            print("---------")
            if "-" in answer or "-" in output:
                correct_format += 1
                
            if final_answer == target or output == target or answer == target:
                match += 1

            # results_to_save.append(
            #     {
            #         "image": result["image"],
            #         "question": result["question"],
            #         "output": output,
            #         "target": target,
            #         "answer": answer,
            #         "correct": answer == target or output == target,
            #     }
            # )

    inference_folder = "/".join(inference_result.split("/")[:-1])

    accuracy = match / len(data) * 100
    correct_format = correct_format / len(data) * 100

    # Save y_true and y_pred for debugging
    with open(f"{inference_folder}/y_true.json", "w") as f:
        json.dump(y_true, f, indent=4)
    with open(f"{inference_folder}/y_pred.json", "w") as f:
        json.dump(y_pred, f, indent=4)

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # --- Assume y_true and y_pred are your data lists ---

    # 1. Define your labels explicitly
    # The labels that are actually correct (True classes)
    true_labels = sorted(list(np.unique(y_true)))
    # All possible outputs, including the invalid ones (Predicted classes)
    all_labels = true_labels + ["invalid"]

    # 2. Calculate the matrix using the complete list of labels
    # This ensures 'Invalid' predictions are counted in a new column
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # 3. Slice the matrix to remove the "Invalid" row
    # Since "Invalid" is not a true class, its row in the matrix will be all zeros.
    # We select all rows corresponding to the true_labels and all columns.
    cm = cm[: len(true_labels), :]

    # 4. Plot the heatmap with correct labels
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        # xticklabels should show all possible predictions, including 'Invalid'
        xticklabels=all_labels,
        # yticklabels should only show the true classes
        yticklabels=true_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (with Invalid Class)")
    # Make sure to create the folder if it doesn't exist
    import os

    os.makedirs(inference_folder, exist_ok=True)
    plt.savefig(f"{inference_folder}/confusion_matrix.png")
    print(f"Confusion matrix saved to {inference_folder}/confusion_matrix.png")
    plt.show()  # Using show() for demonstration

    from sklearn.metrics import f1_score
    import numpy as np

    # --- Assume y_true and y_pred are your data lists ---

    # 1. Define the list of true labels you want to evaluate
    # true_labels = sorted(list(np.unique(y_true)))

    # 2. Calculate F1 scores using the 'labels' parameter
    # This forces the calculation to be based only on the true classes.
    # Predictions of "Invalid" will correctly count as False Negatives for them.
    f1_macro = (
        f1_score(
            y_true,
            y_pred,
            labels=true_labels,  # <-- The crucial fix
            average="macro",
            zero_division=0,
        )
        * 100
    )

    f1_weighted = (
        f1_score(
            y_true,
            y_pred,
            labels=true_labels,  # <-- The crucial fix
            average="weighted",
            zero_division=0,
        )
        * 100
    )

    result = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "correct_format": correct_format,
        "num_samples": len(data),
    }

    with open(f"{inference_folder}/evaluation.json", "w") as f:
        json.dump(result, f, indent=4)

    # with open(f"{inference_folder}/detailed_results.json", "w") as f:
    #     json.dump(results_to_save, f, indent=4)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score (Macro): {f1_macro:.2f}%")  # <-- ADDED
    print(f"F1 Score (Weighted): {f1_weighted:.2f}%")  # <-- ADDED
    print(f"Correct Format: {correct_format:.2f}%")
