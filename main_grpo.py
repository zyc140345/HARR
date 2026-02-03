from dotenv import load_dotenv
load_dotenv()

import importlib
import os
import sys
import datasets as ds
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModel, AutoTokenizer

from trl import ModelConfig, ScriptArguments, get_peft_config
from trl.scripts.utils import TrlParser
from trl_hacked.trainer.grpo_trainer import RetrieverGRPOTrainer
from trl_hacked.trainer.grpo_config import RetrieverGRPOConfig
from reward import (
    em_reward,
    f1_reward,
    sub_em_reward,
    rouge_reward,
    llm_correctness_reward,
    relevance_reward,
    novelty_reward,
)


reward_funcs_registry = {
    "rouge": rouge_reward,
    "em": em_reward,
    "f1": f1_reward,
    "sub_em": sub_em_reward,
    "llm_correctness": llm_correctness_reward,
    "relevance": relevance_reward,
    "novelty": novelty_reward,
}


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the RetrieverGRPO training script."""

    reward_funcs: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": f"Reward functions to use. Choose from {list(reward_funcs_registry.keys())}; or a dotted "
            "import path. (e.g., 'my_lib.rewards.custom_reward')."
        },
    )


def get_run_name(model_args, script_args, training_args):
    model_name = model_args.model_name_or_path.split("/")[-1]
    dataset_name = script_args.dataset_name
    rag_method = training_args.rag_method
    ablation = "|ablation" if not training_args.enable_stateful else ""
    return f"{model_name}|{dataset_name}|{rag_method}{ablation}"


def get_dataset_repo_id(model_args, script_args, training_args):
    dataset_name = script_args.dataset_name
    if dataset_name.endswith("filtered"):
        import huggingface_hub
        hf_username = huggingface_hub.whoami()["name"]
        model_name = model_args.model_name_or_path.split("/")[-1].lower()
        rag_method = training_args.rag_method.replace("_", "-")
        ablation = "ablation_" if not training_args.enable_stateful else ""
        return f"{hf_username}/{model_name}_{rag_method}_{ablation}filtered"
    else:
        return "RUC-NLPIR/FlashRAG_datasets"


def main(script_args, training_args, model_args):
    training_args.run_name = get_run_name(model_args, script_args, training_args)

    # Load a pretrained model
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    # Get the reward functions
    reward_funcs = []
    for func_name in script_args.reward_funcs or ["rouge"]:
        if func_name in reward_funcs_registry:
            reward_funcs.append(reward_funcs_registry[func_name])
        elif "." in func_name:
            module_path, func_name = func_name.rsplit(".", 1)
            sys.path.insert(0, os.getcwd())
            module = importlib.import_module(module_path)
            reward_func = getattr(module, func_name)
            reward_funcs.append(reward_func)
        else:
            raise ValueError(
                f"Could not load reward function '{func_name}'. Expected one of "
                f"{list(reward_funcs_registry.keys())} or a valid import path."
            )

    # Load the dataset
    dataset_repo_id = get_dataset_repo_id(model_args, script_args, training_args)
    if script_args.dataset_name == "all_filtered":
        dataset = ds.concatenate_datasets([
            ds.load_dataset(dataset_repo_id, subset, split=script_args.dataset_train_split)
            for subset in ["hotpotqa", "2wikimultihopqa", "nq", "triviaqa"]
        ])
    elif script_args.dataset_name == "all":
        dataset = ds.concatenate_datasets([
            ds.load_dataset(dataset_repo_id, subset, split=script_args.dataset_train_split)
            for subset in ["hotpotqa", "2wikimultihopqa", "nq", "triviaqa"]
        ])
    elif script_args.dataset_name.endswith("filtered"):
        dataset = ds.load_dataset(
            dataset_repo_id, script_args.dataset_name.split("_")[0],
            split=script_args.dataset_train_split
        )
    else:
        dataset = ds.load_dataset(
            dataset_repo_id, script_args.dataset_name, split=script_args.dataset_train_split
        )
    dataset = dataset.rename_columns({"question": "query", "golden_answers": "ground_truths"})
    eval_samples = 16
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    # Initialize the RetrieverGRPO trainer
    trainer = RetrieverGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser():
    dataclass_types = (GRPOScriptArguments, RetrieverGRPOConfig, ModelConfig)
    return TrlParser(dataclass_types)


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
