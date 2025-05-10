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
"""
Preprocess the GSM8k dataset to parquet format
"""
import sys
sys.path.append('/home/projects/polyullm/yuyang/train_codes/verl-tool')
from pathlib import Path
import datasets
import os
import fire
import numpy as np
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string



execution_prompt = """\
Answer the given coding question. You must conduct reasoning inside <think> and </think> first before you can finally output the final program. During the thinking, you can test your program by writing it inside ```python and ``` tags following with "```output". The code will be executed, and the terminal output (standard output and standard error) will be returned between <output> and </output>. Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. If you find no further code execution needed, you can then give the final program in a markdown code block like this: ```python\nyour code here\n``` without appending anything,. The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""

naive_instruction = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."
naive_execution_prompt = """\
Let's think step by step and generate the correct program for this coding question. You are able to run the python code in the markdown code block and the output will be provided to you in the ````output` block. Put your final program in a markdown code block like this: ```python\nyour code here\n```.
"""

coder_instruction = """\
Let's think step by step and generate the correct program for this coding question. You should attempt multiple times before give the final program.
In each attempt, you should 
- test your program by reviewing the code syntax and logic, and fix any potential issues in the next attempt.
- imagine a set of test cases based on your understanding of the problem and the constraints. 
- You then need to test your program with these test cases. Since you are not able to run the program in a real environment, you need to use text to simulate the program running and think loudly to describe how each variable changes during the execution. Finally, see whether the program produces the expected output.
- if the program fails any of the test cases, you need to debug the program and fix the issues in the next attempt.
- if the program passes all the test cases, you can then give the final program in a markdown code block like this: ```python\nyour code here\n```.

You are also allowed to analyze the problem with any other domain-specific knowledge you have, like math, physics, etc to help you solve the problem.

Now start thinking and generate the final program in a markdown code block like this: ```python\nyour code here\n```.
"""

math_system_prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.
'''

mathcoder_system_prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. For math problems, please put your final answer within \\boxed{}. For code problems, please put your final answer in a markdown code block like this: ```python\nyour code here\n```.
'''

### Utils ###
def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


### Main Preprocessing Function ###
def main(
    local_dataset_path: str,
    local_dir: str = 'data/mathcoder',
    hdfs_dir: str = None,
    level: str = 'hard',
    add_execution_prompt: bool = False,
    detaield_instruction: bool = False
):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    _process_local_math(local_dataset_path, local_dir)


### Math Dataset Logic ###
def _process_local_math(data_source, local_dir):
    print(f"Loading the {data_source} dataset...", flush=True)
    dataset = datasets.load_dataset('parquet', data_dir=data_source)
    data_source = data_source.rsplit('/', 1)[-1]
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('solution')
            solution = extract_solution(answer)
            
            data = {
                "data_source": data_source,
                "prompt": [
                {
                    "role": "system",
                    "content": mathcoder_system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test', data_source), with_indices=True, remove_columns=test_dataset.column_names)

    print(train_dataset)
    print(train_dataset[0])

    train_dataset.to_parquet(os.path.join(local_dir, f'{data_source}_train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, f'{data_source}_test.parquet'))
    

if __name__ == '__main__':
    fire.Fire(main)

"""
python examples/data_preprocess/mathcoder_local.py --local_dataset_path ../../datasets/NuminaMath-TIR --local_dir data/NuminaMath-TIR
"""
