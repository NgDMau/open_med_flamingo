import re
import json
import argparse
from tqdm import tqdm


def extract_answer_regex(text: str) -> str:
    """
    Extracts a hyphenated word following 'Final Answer: ' using regex.

    Args:
        text: The input string to search.

    Returns:
        The extracted word (e.g., 'Mild-Dementia') or None if no match is found.
    """
    # This pattern looks for "Final Answer:", followed by optional whitespace,
    # and then captures a sequence of word characters and hyphens.
    match = re.search(r"Final Answer:\s*([\w-]+)", text)
    
    if match:
        return match.group(1) # Return the first captured group
    return "None"


if __name__ == "__main__":

    default_inference_result = "/mnt/data/nict/maund/baseline_models/instruct_flamingo/predictions_validation/0821-clever_flamingo_v2_3b-2k_context-40G-resume-from-mpt7b-checkpoint-03-checkpoint_15/eval_dataset_config_-1/llava-mri-cot-1k-test_all.json"

    # make inference_result path a command line argument: --inference_result
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_result", type=str, default=default_inference_result)
    args = parser.parse_args()
    
    inference_result = args.inference_result
    
    with open(inference_result, "r") as f:
        data = json.load(f)

    correct_format = 0
    match = 0

    for result in tqdm(data):
        output = result["output"]
        target = result["target"]

        answer = extract_answer_regex(output)
        if answer and target:
            answer = answer.lower().replace(" ", "")
            target = target.lower().replace(" ", "")

            print("---------")
            if "-" in answer:
                correct_format += 1
            if (not (answer == target)) and ("-" not in answer):
                print(f"Output: {output}")
                print(f"Answer: {answer}")
                print(f"Target: {target}")
            if answer == target:
                match += 1
            
    # Get inference folder from inference_result path
    inference_folder = "/".join(inference_result.split("/")[:-1])

    accuracy = match / len(data) * 100
    correct_format = correct_format / len(data) * 100
    
    result = {
        "accuracy": accuracy,
        "correct_format": correct_format,
        "num_samples": len(data)
    }
    
    with open(f"{inference_folder}/evaluation.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"Accuracy: {accuracy}%")
    print(f"Correct Format: {correct_format}%")