import json
import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import logging
import re
from typing import List, Tuple
from collections import defaultdict
from data import InstructionDataset
from inferencer import Inferencer

parser = argparse.ArgumentParser()

# Build Inferencer - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--tuning_config", default=None, type=str)
parser.add_argument("--checkpoint_paths", type=str, default=None)
parser.add_argument(
    "--cross_attn_every_n_layers",
    type=int,
    default=4,
    help="how often to add a cross-attention layer after each transformer layer",
)
parser.add_argument("--v1", action="store_true", default=False)

# Language Generation Configs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
parser.add_argument("--max_new_token", type=int, default=128)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_k", type=float, default=20)
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument("--do_sample", type=bool, default=False)
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
parser.add_argument("--length_penalty", type=float, default=1)
parser.add_argument("--max_length", type=int, default=1024)

# Add batch_size as arg
parser.add_argument("--batch_size", type=int, default=8)

# Dataset Configs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
parser.add_argument(
    "--results_dir", type=str, default=None, help="JSON file to save results"
)
parser.add_argument(
    "--instruction_path",
    type=str,
    help="Path to the instruction dataset directory.",
    default=None,
)
parser.add_argument(
    "--img_dir",
    type=str,
    help="Path to the instruction dataset images.",
    default=None,
)
parser.add_argument("--instruction_prompt_templete", type=str, default="guanaco")
parser.add_argument("--dataset_sampling_mode", type=str, default="ratio")
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)


def add_image_dir(str, img_dir):
    if img_dir == "":
        return str
    else:
        img_path_count = 0
        index = 0
        while index < len(str):
            if str.startswith("<img_path>", index):
                img_path_count += 1
                if img_path_count % 2 == 1:  # Check if it's an odd-numbered <img_path>
                    str = (
                        str[:index]
                        + "<img_path>"
                        + img_dir
                        + "/"
                        + str[index + len("<img_path>") :]
                    )
            index += 1
        return str


def save_results(results, args, logger, rank):
    if not os.path.exists(args.results_dir):
        logger.info(f"Creating results directory at {args.results_dir}")
        os.makedirs(args.results_dir)
    all_results = []
    for dataset_name, dataset_results in results.items():
        logger.info(
            f"Saving results for {dataset_name} to file {args.results_dir}/{dataset_name}.json"
        )
        all_results.extend(dataset_results)
        with open(
            os.path.join(args.results_dir, f"{dataset_name}_rank{rank}.json"), "w"
        ) as f:
            json.dump(dataset_results, f, indent=4)
    # with open(os.path.join(args.results_dir, f"all_results.json"), "w") as f:
    #     json.dump(all_results, f, indent=4)


def save_summary(results, args, logger):
    summary = defaultdict(dict)
    header = "| Dataset | Avg ACCURACY |Avg target length | Avg prediction length | Exact matches | Match percentage | Total samples |\n|---------|---------- |-----------------|---------------------|---------------|-----------------|---------------|\n"

    for dataset_name, dataset_results in results.items():
        references = [result["target"] for result in dataset_results]
        hypotheses = [result["output"] for result in dataset_results]

        references, hypotheses = preprocess_text(references, hypotheses)

        logger.info(f"Calculating metrics for {dataset_name}...")
        accuracy = calculate_accuracy(
            references=references, hypotheses=hypotheses, loose=True
        )

        target_lengths = [len(target.split()) for target in references]
        prediction_lengths = [len(prediction.split()) for prediction in hypotheses]

        exact_matches = sum(
            [target == prediction for target, prediction in zip(references, hypotheses)]
        )

        summary[dataset_name] = {
            "accuracy": accuracy,
            "avg_target_length": (
                sum(target_lengths) / len(target_lengths) if target_lengths else 0
            ),
            "avg_prediction_length": (
                sum(prediction_lengths) / len(prediction_lengths)
                if prediction_lengths
                else 0
            ),
            "exact_matches": exact_matches,
            "match_percentage": (
                (exact_matches / len(references) * 100) if references else 0
            ),
            "total_samples": len(references),
        }

        header += f"| {dataset_name} | {accuracy:.2f} | {summary[dataset_name]['avg_target_length']:.2f} | {summary[dataset_name]['avg_prediction_length']:.2f} | {exact_matches} | {summary[dataset_name]['match_percentage']:.2f}% | {len(references)} |\n"

    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(header)


def preprocess_text(
    references: List[str], hypotheses: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Preprocesses lists of reference and hypothesis strings for evaluation.

    This function applies the following steps to each string:
    1. Converts the string to lowercase.
    2. Removes leading and trailing whitespace.
    3. Replaces one or more internal spaces with a single hyphen.

    Args:
        references: A list of ground truth strings.
        hypotheses: A list of strings generated by the model.

    Returns:
        A tuple containing two lists: the processed references and processed hypotheses.
    """
    processed_references = []
    for ref in references:
        # 1. Lowercase and strip whitespace
        processed_ref = ref.lower().strip()
        # 2. Replace internal spaces with a hyphen
        processed_ref = re.sub(r"\s+", "-", processed_ref)
        processed_references.append(processed_ref)

    processed_hypotheses = []
    for hyp in hypotheses:
        # 1. Lowercase and strip whitespace
        processed_hyp = hyp.lower().strip()
        # 2. Replace internal spaces with a hyphen
        processed_hyp = re.sub(r"\s+", "-", processed_hyp)
        processed_hypotheses.append(processed_hyp)

    return processed_references, processed_hypotheses


def calculate_accuracy(
    references: List[str], hypotheses: List[str], loose=False
) -> float:
    """
    Calculates the accuracy (exact match) score between two lists of strings.

    Args:
        references: A list of preprocessed ground truth strings.
        hypotheses: A list of preprocessed strings generated by the model.

    Returns:
        The accuracy score as a float (from 0.0 to 1.0).
        Returns 0.0 if the lists are empty or have different lengths.
    """
    # Ensure lists are not empty and have the same length
    if not references or len(references) != len(hypotheses):
        print("Error: Input lists must not be empty and must have the same length.")
        return 0.0

    correct_predictions = 0
    total_predictions = len(references)
    for ref, hyp in zip(references, hypotheses):
        if ref == hyp:
            correct_predictions += 1
        elif loose:
            # Check if there is pattern <Final Answer: Some-thing> inside hyp
            # For example: "Final Answer: Mild-Dementia"
            answer = extract_answer_regex(hyp)
            if ref == answer:
                correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy


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
        return match.group(1)  # Return the first captured group
    return ""


def main():
    # Distributed initialization
    print("Distribution Initialization...")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print("World Size:", world_size)
    print(f"Local Rank: {local_rank}, Rank: {rank}")

    print("Setting rank...")
    torch.cuda.set_device(local_rank)

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [%(levelname)s] [RANK {rank}] %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    args.rank = rank

    def random_seed(seed=42, rank=0):
        torch.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)

    random_seed(args.seed, rank)

    if args.checkpoint_paths is not None:
        args.ckpt_basename = (
            os.path.dirname(args.checkpoint_paths).split("/")[-1]
            + "-"
            + os.path.basename(args.checkpoint_paths).replace(".pt", "")
        )
    else:
        args.ckpt_basename = "no_checkpoint"
    args.results_dir = os.path.join(
        args.results_dir,
        args.ckpt_basename,
        f'{os.path.basename(args.instruction_path).replace(".json", "")}_{args.num_samples}',
    )
    if rank == 0:
        logger.info("args " + "-" * 100)
        for key, value in args.__dict__.items():
            logger.info("\t{:<30}\t{}".format(key + ":", value))
        logger.info("-" * 100)

    print("Initialization Complete")

    # ------------------------------------
    # Load Model and Checkpoints
    # ------------------------------------

    print("Load model and checkpoint...")

    inferencer = Inferencer(
        lm_path=args.lm_path,
        checkpoint_paths=args.checkpoint_paths,
        tuning_config=args.tuning_config,
        clip_vision_encoder_path=args.vision_encoder_path,
        clip_vision_encoder_pretrained=args.vision_encoder_pretrained,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        v1=args.v1,
        device=f"cuda:{local_rank}",
    )

    # ------------------------------------
    # Load Datasets and Runs Inference
    # ------------------------------------

    dataset = InstructionDataset(
        config_json_path=args.instruction_path,
        image_processor=inferencer.image_processor,
        tokenizer=inferencer.tokenizer,
        num_samples=args.num_samples,
        max_length=None,
        logger=logger,
        img_dir=args.img_dir,
        args=args,
        mode="test",
    )

    dataset_names = []
    dataset_results = defaultdict(list)

    batch_size = args.batch_size  # You can adjust this value as needed
    print("Using batch size:", batch_size)
    batch_size = 8
    print("Change to batch size:", batch_size)
    batch_prompts = []
    batch_img_paths = []
    batch_samples = []
    batch_indices = []

    # Shard dataset by rank for data parallelism
    total_len = len(dataset)
    indices = list(range(total_len))
    indices = indices[rank::world_size]

    for count, index in enumerate(tqdm(indices) if rank == 0 else indices):
        item = dataset[index]
        img_paths, text, instruction_str, sample = item

        # Check if all image paths exist for this sample
        all_exist = True
        for img_path in img_paths:
            if not os.path.exists(img_path):
                logger.warning(
                    f"Image path does not exist: {img_path}. Skipping index {index}."
                )
                all_exist = False
                break
        if not all_exist:
            continue

        batch_prompts.append(instruction_str)
        batch_img_paths.append(img_paths)
        batch_samples.append(sample)
        batch_indices.append(index)
        
        # --- Add this block to check token length ---
        # Count text tokens
        text_token_counts = [len(inferencer.tokenizer(prompt)["input_ids"]) for prompt in batch_prompts]
        # Count image tokens (assuming each <image> token is a special token in the prompt)
        image_token_counts = [prompt.count("<image>") for prompt in batch_prompts]
        # Total tokens (text + image)
        total_token_counts = [t + i for t, i in zip(text_token_counts, image_token_counts)]
        for idx, (t, i, total) in enumerate(zip(text_token_counts, image_token_counts, total_token_counts)):
            logger.info(f"Sample {batch_indices[idx]}: text tokens={t}, image tokens={i}, total tokens={total}")
        # --- End block ---

        # If batch is full or last sample, run inference
        if len(batch_prompts) == batch_size or count == len(indices) - 1:
            predictions, full_texts = inferencer(
                prompt=batch_prompts,
                images=batch_img_paths,
                max_new_token=args.max_new_token,
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                response_split="### Assistant:",
            )
            for i, (prediction, sample, prompt, idx) in enumerate(
                zip(predictions, batch_samples, batch_prompts, batch_indices)
            ):
                dataset_name = dataset.configs[sample["dataset_idx"]]["dataset_name"]
                prompt_clean = prompt.replace("<|endofchunk|>", "")
                dataset_results[dataset_name].append(
                    {
                        "input": add_image_dir(sample["input"], sample["img_dir"]),
                        "output": prediction,
                        "target": sample["output"],
                        "prompt": prompt_clean,
                    }
                )
                dataset_names.append(dataset_name)

            # Save results every 10 samples (not every batch)
            if count % 10 == 0:
                save_results(dataset_results, args, logger, rank)

            # Reset batch
            batch_prompts = []
            batch_img_paths = []
            batch_samples = []
            batch_indices = []

    # Save per-rank results
    save_results(dataset_results, args, logger, rank)

    # Optionally, gather results to rank 0 for summary
    dist.barrier()
    if rank == 0:
        # Merge all per-rank result files
        merged_results = defaultdict(list)
        for r in range(world_size):
            for dataset_name in dataset_results.keys():
                result_file = os.path.join(
                    args.results_dir, f"{dataset_name}_rank{r}.json"
                )
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        merged_results[dataset_name].extend(json.load(f))
        # Save merged results
        for dataset_name, results in merged_results.items():
            with open(
                os.path.join(args.results_dir, f"{dataset_name}_all.json"), "w"
            ) as f:
                json.dump(results, f, indent=4)
        # Compute summary on merged results
        save_summary(merged_results, args, logger)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
