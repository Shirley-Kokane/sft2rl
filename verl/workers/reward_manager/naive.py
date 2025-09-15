# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from nltk.translate.bleu_score import sentence_bleu


def diversity_score(responses: list[str], response_ids: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """
    Compute the diversity score for a batch of responses.
    """
    # Get the unique indices
    prompt_to_indices = defaultdict(list)
    for idx, prompt in enumerate(indices):
        prompt_to_indices[prompt].append(idx)
    
    # Calculate the diversity score
    index_to_rows = {prompt: [] for prompt in indices}
    index_to_pos = {prompt: [] for prompt in indices}
    index_to_token_ids = {prompt: [] for prompt in indices}
    index_to_set = {prompt: set() for prompt in indices}

    for i, prompt in enumerate(indices):
        max_tokens = min(len(responses[i]), 50)
        index_to_rows[prompt].append(responses[i][:max_tokens])
        index_to_token_ids[prompt].append(response_ids[i][:max_tokens].tolist())
        index_to_pos[prompt].append(i)  # to maintain position of each rollout
    
    import itertools
    for prompt in index_to_rows:
        index_to_set[prompt] = set(itertools.chain.from_iterable(index_to_token_ids[prompt]))
        
    # print("how many rollouts ", len(index_to_rows.keys()) , len(index_to_rows[prompt]))
    self_bleu_scores = [0.0]*len(response_ids)
    per_rollout_uniqueness = [0.0] * len(response_ids)  # aligned with input
        
    for prompt in indices:
        rollouts = index_to_rows[prompt]
        token_ids = index_to_token_ids[prompt]
        
        # Self-BLEU
        i=0
        if len(rollouts) < 2:
            self_bleu_scores[index_to_pos[prompt][i]] = 0.0
            per_rollout_uniqueness[index_to_pos[prompt][i]] = 0.0
        else:
            for i in range(len(rollouts)):
                references = rollouts[:i] + rollouts[i+1:]
                score = sentence_bleu(references, rollouts[i])
                self_bleu_scores[index_to_pos[prompt][i]] = score
                
                reference_ids = token_ids[:i] + token_ids[i+1:]
                reference_set = set(itertools.chain.from_iterable(reference_ids))
                current_set = set(token_ids[i])
                unique_tokens = reference_set - current_set
                per_rollout_uniqueness[index_to_pos[prompt][i]] = len(unique_tokens) / len(reference_set)
                #print("tokens ", unique_tokens, reference_set, current_set, per_rollout_uniqueness[index_to_pos[prompt][i]])
    
    return self_bleu_scores, per_rollout_uniqueness


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        indices = [item["index"] for item in data.non_tensor_batch["extra_info"]]
        #prompts = [item["prompt"] for item in data.non_tensor_batch["extra_info"]]
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        responses = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses.append(response_str.split(" "))

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        
        self_bleu_scores, per_rollout_uniqueness = diversity_score(responses, data.batch["responses"], indices)
        
        reward_extra_info["self_bleu_scores"] = self_bleu_scores
        reward_extra_info["per_rollout_uniqueness"] = per_rollout_uniqueness

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
