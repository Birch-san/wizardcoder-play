# WizardCoder-Play

<img height="250" alt="image" src="https://github.com/Birch-san/wizardcoder-play/assets/6141784/e0292ebf-8998-460b-a021-a4c1b5511d20">

Python script to demonstrate how to invoke models such as WizardCoder from the command-line, with bitsandbytes 4-bit quantization.

Intends to support the following models:

- [`WizardLM/WizardCoder-Python-7B-V1.0`](https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0)
- [`WizardLM/WizardCoder-Python-13B-V1.0`](https://huggingface.co/WizardLM/WizardCoder-Python-13B-V1.0)
- [`WizardLM/WizardCoder-Python-34B-V1.0`](https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0)
- [`codellama/CodeLlama-7b-Instruct-hf`](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- [`codellama/CodeLlama-13b-Instruct-hf`](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf)
- [`codellama/CodeLlama-34b-Instruct-hf`](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)

CodeLlama models were [trained on 16000 token sequences](https://ai.meta.com/blog/code-llama-large-language-model-coding/).  
WizardCoder was [finetuned on 2048 token sequences](https://arxiv.org/abs/2306.08568).

WizardCoder-Python-34B-V1.0 [surpasses](https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0) GPT4, ChatGPT-3.5 and Claude2 on HumanEval benchmarks.

## Setup

All instructions are written assuming your command-line shell is bash.

Clone repository:

```bash
git clone https://github.com/Birch-san/wizardcoder-play.git
cd wizardcoder-play
```

### Create + activate a new virtual environment

This is to avoid interfering with your current Python environment (other Python scripts on your computer might not appreciate it if you update a bunch of packages they were relying on).

Follow the instructions for virtualenv, or conda, or neither (if you don't care what happens to other Python scripts on your computer).

#### Using `venv`

**Create environment**:

```bash
python -m venv venv
pip install --upgrade pip
```

**Activate environment**:

```bash
. ./venv/bin/activate
```

**(First-time) update environment's `pip`**:

```bash
pip install --upgrade pip
```

#### Using `conda`

**Download [conda](https://www.anaconda.com/products/distribution).**

_Skip this step if you already have conda._

**Install conda**:

_Skip this step if you already have conda._

Assuming you're using a `bash` shell:

```bash
# Linux installs Anaconda via this shell script. Mac installs by running a .pkg installer.
bash Anaconda-latest-Linux-x86_64.sh
# this step probably works on both Linux and Mac.
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda config --set auto_activate_base false
conda init
```

**Create environment**:

```bash
conda create -n p311-llama python=3.11
```

**Activate environment**:

```bash
conda activate p311-llama
```

### Install package dependencies

**Ensure you have activated the environment you created above.**

Install dependencies:

```bash
pip install -r requirements.txt
```

#### (Optional) install PyTorch nightly

The PyTorch nightlies may be more performant. Until [PyTorch 2.1.0 stable comes out (~October 4th)](https://github.com/pytorch/pytorch/issues/86566#issuecomment-1706075651), nightlies are the best way to get CUDA 12.1 support:

```bash
# CUDA
pip install --upgrade --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu121
```

#### (Optional) install flash attention 2

To accelerate inference and reduce memory usage, install `flash-attn`.

First we install the package itself:

```bash
pip install flash-attn --no-build-isolation
```

Then we build-from-source its rotary embeddings kernel (there is no officially-distributed wheel):

```bash
MAX_JOBS=2 pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

**[Building `rotary` from source] `error: expected template-name before ‘<’ token`:**  
If you compiled flash-attn source using nvcc 12.x (i.e. CUDA Toolkit 12), you will [encounter the following error](https://github.com/pybind/pybind11/issues/4606) whilst compiling pybind11's `cast.h` header:

```
/home/birch/anaconda3/envs/p311-cu121-bnb-opt/lib/python3.11/site-packages/torch/include/pybind11/detail/../cast.h: In function ‘typename pybind11::detail::type_caster<typename pybind11::detail::intrinsic_type<T>::type>::cast_op_type<T> pybind11::detail::cast_op(make_caster<T>&)’:
/home/birch/anaconda3/envs/p311-cu121-bnb-opt/lib/python3.11/site-packages/torch/include/pybind11/detail/../cast.h:45:120: error: expected template-name before ‘<’ token
   45 |     return caster.operator typename make_caster<T>::template cast_op_type<T>();
```

Solution [here](https://github.com/Dao-AILab/flash-attention/issues/484#issuecomment-1706843478).

## Run:

From root of repository:

```bash
python -m scripts.wizard_play
```

Fun command-line options:

- `--model_name_or_path WizardLM/WizardCoder-Python-7B-V1.0 --prompt_style wizardcoder-python`: use WizardCoder 7B with WizardCoder prompting style
- `--model_name_or_path codellama/CodeLlama-7b-Instruct-hf --prompt_style codellama-instruct`: use CodeLlama-7b-Instruct with CodeLlama-Instruct prompting style
- `--flash --trust_remote_code`: enables flash attention 2 via `flash-attn` library and ([my fork of](https://huggingface.co/Birchlabs/flash_llama)) [togethercomputer's `modeling_flash_llama.py`](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/modeling_flash_llama.py)
- `--max_new_tokens 2048`: modify maximum response length
- `--chat_memory`: enable conversation history, for multi-turn conversations (CodeLlama-Instruct was trained on this, but WizardCoder was not)
- `--initial_input 'Write a function which computes the Fibonacci sequence.'`: you can buffer a prompt to be submitted as soon as the model's loaded.

You can press Ctrl+C whilst the model is generating a response, to interrupt it. If `--chat_memory` is enabled: the unfinished message **does** get persisted into the conversation history.  
If the model is **not** generating a response, then Ctrl+C will exit the software.

### Few-shotting

You can seed the conversation history with a previous input and forced response from the model:

```bash
python -m scripts.wizard_play --model_name_or_path codellama/CodeLlama-7b-Instruct-hf --prompt_style codellama-instruct --shot0_input "Read user's name from stdin" --shot0_response 'import sys

name = input("Enter your name: ")
print("Your name is:", name)'
```

This achieves two things:

- creates a memory in the conversation
- sets an expectation for what kind of style of response you prefer.

You can see this in action, by asking the model to iterate on the solution you placed into its history:

```
[seed=64]$ Print their age too.
import sys

name = input("Enter your name: ")
age = input("Enter your age: ")
print("Your name is:", name, ",", "and", "your age:", age)
```

Note: this won't necessarily work so well for WizardCoder, which isn't trained in multi-turn conversations.

### Troubleshooting

**`cannot import name 'translate_llvmir_to_hsaco'`:**  
You [need a triton nightly](https://github.com/openai/triton/issues/2002).
```
Failed to import transformers.models.llama.modeling_llama because of the following error (look up to see its traceback):
Failed to import transformers.generation.utils because of the following error (look up to see its traceback):
cannot import name 'translate_llvmir_to_hsaco' from 'triton._C.libtriton.triton' (unknown location)
```

```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

**`ImportError`:**  
Recent flash-attn releases encounter [errors _importing_ rotary embed](https://github.com/Dao-AILab/flash-attention/issues/519). You may need to copy Dao-AILab's [`ops/triton`](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/ops/triton) directory into the flash-attn distribution you installed to site-packages.

## License

This repository is itself MIT-licensed.

Includes:

- MIT-licensed code copied from Artidoro Pagnoni's [qlora](https://github.com/artidoro/qlora)
- MIT-licensed code copied from Scott Logic's [qlora fork](https://github.com/scottlogic-alex/qlora) (specifically [`evaluate.py`](https://github.com/scottlogic-alex/qlora/blob/stepwise/evaluate.py)).
- [Apache-licensed](licenses/MosaicML-mpt-7b-chat-hf-space.Apache.LICENSE.txt) code copied from MosaicML's [mpt-7b-chat](https://huggingface.co/spaces/mosaicml/mpt-7b-chat/blob/main/app.py) Huggingface Space
