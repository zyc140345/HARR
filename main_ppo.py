# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dotenv import load_dotenv
load_dotenv()

import shutil

import torch
import datasets as ds
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl_hacked.trainer.ppo_trainer import RetrieverPPOTrainer
from trl_hacked.trainer.ppo_config import RetrieverPPOConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from reward import (
    em_reward,
    f1_reward,
    sub_em_reward,
    rouge_reward,
    llm_correctness_reward,
    relevance_reward,
    novelty_reward,
)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RetrieverPPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModel.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "rouge": rouge_reward,
        "em": em_reward,
        "f1": f1_reward,
        "sub_em": sub_em_reward,
        "llm_correctness": llm_correctness_reward,
        "relevance": relevance_reward,
        "novelty": novelty_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in training_args.reward_funcs]

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModel.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = ds.load_dataset("RUC-NLPIR/FlashRAG_datasets", script_args.dataset_name, split="train")
    dataset = dataset.rename_columns({"question": "query", "golden_answers": "ground_truths"})
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    def data_collator(batch):
        return {
            k: [example[k] for example in batch]
            for k in batch[0].keys()
        }

    ################
    # Training
    ################
    trainer = RetrieverPPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_funcs=reward_funcs,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
