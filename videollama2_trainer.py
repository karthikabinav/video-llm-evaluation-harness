# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import re
import os
import copy
import json
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import Dataset

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import sys
sys.path.append("./")

# VideoLLaMA2 training implementation
# Full trainer code from DAMO-NLP-SG/VideoLLaMA2 repository