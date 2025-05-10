# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
import regex as re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
import time
import torch

from verl import DataProto
from .reward_score import _default_compute_score


import hashlib
import random
import os
import json
import subprocess
import time
from pathlib import Path

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()


async def single_compute_score(evaluation_func, completion, reference, task, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, executor, timeout=300.)
            for completion, reference, task in zip(completions, references, tasks)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


def parse_code(action: str):
    """
    Parse the raw action string (which is the llm response) into an actual action and its contents.
    Ensures that the parsed code is valid and safe for execution.
    
    Args:
        action: Raw action string containing Python code
        
    Returns:
        Tuple containing the extracted code and a validity flag
    """
    # Try to find Python code in various formats
    all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
    
    if not all_valid_python_code:
        all_valid_python_code = re.findall(r"```python(.*?)```", action, re.DOTALL)
    
    if not all_valid_python_code:
        all_valid_python_code = re.findall(r"```(.*?)```", action, re.DOTALL)
    
    if len(all_valid_python_code) == 0:
        return ""
    
    # Use the final code block found (we could extend this to support multiple blocks)
    parsed_code = all_valid_python_code[-1].strip()
    
    return parsed_code

class AceCoderRewardManager:
    """
    The Reward Manager used in https://github.com/TIGER-AI-Lab/AceCoder
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, run_id=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.step_idx = 0
        self.n_workers = 64
        self.binary = False
        try:
            from acecoder import evaluate_test_cases
        except ImportError:
            raise ImportError("`from acecoder import evaluate_test_cases` failed, please install acecoder to use test_case rule")
        
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"acecoder-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
                
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # TODO: implement new reward computing & statistic mechanism
        scores = [{} for _ in range(len(data))]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        if "turns_stats" in data.non_tensor_batch:
            num_turn = data.non_tensor_batch["turns_stats"]
            num_valid_action = data.non_tensor_batch["valid_action_stats"]
            is_active = data.non_tensor_batch["active_mask"]
            is_done = [not is_active[i] for i in range(len(is_active))]
            
        already_print_data_sources = {}
        
        # retrieve the list of prompt_token_ids and their length
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        # retrieve the list of response ids and their valid length
        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        
        # batch decode the list of responses and prompts
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
       
        print(type(data))
        # retrieve the list of ground truths/test cases
        if isinstance(data[0].non_tensor_batch['reward_model']['ground_truth'], list):
            ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        else:
            ground_truth = [data_item.non_tensor_batch['extra_info']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        
        
        # 1. Testing code on the test cases
        # extract the answer for the list of responses
        extracted_answers = [re.sub(r"<think>(.|\n)*?</think>", "", response) for response in response_str]
        extracted_answers = [parse_code(response) for response in extracted_answers] # get the last code blcok
        question_hashes = [hash_string(question) for question in prompt_str]
        # ensure the length of lists are of the same, avoid Ray error
        assert len(response_str) == len(ground_truth) == len(data_sources)
        # before perform batched scoring: dump the statistics of the list of responses
        samples = [
            {
                'task_id': question_hash,
                'prompt': question,
                'output': answer,
                'original_response': response,
                'tests': list(test_case),
                '_identifier': f"{question_hash}_{i}"
            }
            for i, (question_hash, question, answer, test_case, response) in enumerate(zip(question_hashes, prompt_str, extracted_answers, ground_truth, response_str))
        ]
        # save the dumped samples to a file
        temp_file = self.record_dir / f"step-{self.step_idx}_{hash_string(''.join(question_hashes))}.jsonl"
        self.step_idx += 1
        with open(temp_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        # perform batched scoring for coding score: call the acecoder evaluation script to retrieve the coder part scores
        output_file = Path(temp_file).with_suffix(f".eval_results{'_binary' if self.binary else ''}.jsonl").absolute()
        command = f"python -m acecoder.eval_test_cases --samples {temp_file} --n_workers {self.n_workers} \
            --extract_solution True --output_file {output_file} --test_details {not self.binary} \
            --i_just_wanna_run True --min_time_limit 1 --gt_time_limit_factor 1"
        start = time.time()
        subprocess.run(command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        end = time.time()
        print(f"Step {self.step_idx}: acecoder evaluation script took {end - start:.2f} seconds for {len(samples)} samples.")
        # the script will dump the results into the output_file, read it and parse it as a list
        with open(output_file, "r") as f:
            all_samples_results = [json.loads(x) for x in f]
        pass_rates = [x['eval_results']['pass_rate'] for x in all_samples_results]
        # remove the temp_file and output_file after finish code pass rate computation and result extraction
        try:
            os.remove(temp_file)
            os.remove(output_file)
        except:
            pass
        for i in range(len(scores)):
            scores[i]['pass_rate'] = pass_rates[i]
            scores[i]['binary_pass_rate'] = 1.0 if pass_rates[i] == 1.0 else 0.0
            scores[i]['score'] = 1.0 if pass_rates[i] == 1.0 else -1.0
        
        # 2. Adding additional penalty for errored or timed out
        keywords = ["ERROR:\nTraceback", "Execution timed out"]
        for i, response in enumerate(response_str):
            if any(keyword in response for keyword in keywords):
                # time_out_reward_tensor[i, valid_response_length[i].item() - 1] -= 0.5
                scores[i]['exec_error'] = 1
            else:
                scores[i]['exec_error'] = 0
            
            # 2.1.
            # if scores[i]['exec_error'] == 1:
            #     scores[i]['score'] -= 0.5 # minus 0.5 for execution error
            
            if "turns_stats" in data[i].non_tensor_batch:
                num_turn = data[i].non_tensor_batch["turns_stats"]
                num_valid_action = data[i].non_tensor_batch["valid_action_stats"]
                is_active = data[i].non_tensor_batch["active_mask"]
                is_done = not is_active
                # # 2.2. add additional penalty for the number of turns and valid actions
                # if scores[i]['binary_pass_rate'] == 0.0 and num_valid_action < 1:
                #     scores[i]['score'] -= 0.5
                #     scores[i]['tool_use_penalty'] = 1
                # else:
                #     scores[i]['tool_use_penalty'] = 0
                    
            
        for i, score in enumerate(scores):
            if isinstance(score, dict):
                reward_tensor[i, valid_response_length[i].item() - 1] = score['score']
                for k, v in score.items():
                    reward_extra_info[k].append(v)
            else:
                reward_tensor[i, valid_response_length[i].item() - 1] = score
        
        if save_record:
            # Save the records for each code response sample, which will be reported to wandb
            to_save_records = [
                {
                    "id": data[i].non_tensor_batch['extra_info']['id'] if 'id' in data[i].non_tensor_batch['extra_info'] else None,
                    "data_source": data[i].non_tensor_batch['data_source'],
                    "prompt": samples[i]['prompt'],
                    "response": samples[i]['original_response'],
                    "extracted_code": samples[i]['output'],
                    "ground_truth": samples[i]['tests'],
                    "score": scores[i],
                    'extra_info': data[i].non_tensor_batch.get('extra_info', None),
                }
                for i in range(len(data))
            ]
            if "turns_stats" in data.non_tensor_batch:
                for i in range(len(data)):
                    to_save_records[i]['num_turn'] = data[i].non_tensor_batch["turns_stats"]
                    to_save_records[i]['num_valid_action'] = data[i].non_tensor_batch["valid_action_stats"]
                    to_save_records[i]['is_done'] = not data[i].non_tensor_batch["active_mask"]
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"acecoder-step-val-{self.step_idx}.json"
            else:
                temp_file = self.record_dir / f"acecoder-step-{self.step_idx}.json"
            self.step_idx += 1
            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
        
        if return_dict: 
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
