from dataclasses import dataclass, field
from typing import Optional, TypedDict, NamedTuple, List, Dict, Union, TypeAlias, Literal
import torch
from torch import LongTensor
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  GenerationConfig,
  HfArgumentParser,
  set_seed,
  StoppingCriteria,
  StoppingCriteriaList,
  LlamaForCausalLM,
  LlamaTokenizerFast
)
from src.callback_text_iterator_streamer import CallbackTextIteratorStreamer
import logging
from enum import Enum
import sys
from time import perf_counter

logger = logging.getLogger(__name__)

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

class PromptStyle(Enum):
  Bare = 'bare'
  WizardCoderPython = 'wizardcoder-python'
  CodeLlamaInstruct = 'codellama-instruct'
# I am not proud of this, but when I attempted to specify Enum fields on the arg dataclasses:
# hfparser.parse_args_into_dataclasses() turned the enum instances into string values.
# so we make some types to capture what we're actually going to receive.
PromptStyleLiteral: TypeAlias = Literal['bare', 'wizardcoder-python', 'codellama-instruct']

class Dtype(Enum):
  Bf16 = 'bf16'
  Fp16 = 'fp16'
  Fp32 = 'fp32'
DtypeLiteral: TypeAlias = Literal['bf16', 'fp16', 'fp32']

def reify_dtype(dtype: DtypeLiteral) -> torch.dtype:
  match(dtype):
    case 'bf16':
      return torch.bfloat16
    case 'fp16':
      return torch.float16
    case 'fp32':
      return torch.float32

class Participant(Enum):
  User = 'user'
  Assistant = 'assistant'
  System = 'system'

class Message(NamedTuple):
  participant: Participant
  message: str

@dataclass
class StopOnTokens(StoppingCriteria):
  stop_token_ids: List[int]
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in self.stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False

class SufficientResponse(BaseException): ...

@dataclass
class ModelArguments:
  model_name_or_path: Optional[str] = field(
    default="WizardLM/WizardCoder-Python-7B-V1.0"
  )
  cache_dir: Optional[str] = field(
    default=None,
    metadata={"help": "Which directory to use as your HuggingFace cache. Defaults to ~/.cache/huggingface, probably. Use this if you want to download models to a specific location."}
  )
  trust_remote_code: Optional[bool] = field(
    default=False,
    metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
  )
  double_quant: bool = field(
    default=True,
    metadata={"help": "Compress the quantization statistics through double quantization."}
  )
  quant_type: str = field(
    default="nf4",
    metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
  )
  bits: int = field(
    default=4,
    metadata={"help": "How many bits to use.", "choices": [4, 8, 16, 32]}
  )
  model_dtype: DtypeLiteral = field(
    default=Dtype.Fp16.value,
    metadata={"help": "Compute type of the model. Used for non-quantized computations.", "choices": [p.value for p in Dtype]}
  )
  bnb_compute_dtype: DtypeLiteral = field(
    default=Dtype.Fp16.value,
    metadata={"help": "Compute type used for quantized computations. Prefer to turn this on if you are quantizing and your GPU supports it. You probably also want it even if you're not quantizing. Float16 should be better than bfloat16. Float32 can be slightly better than float16.", "choices": [p.value for p in Dtype]}
  )
  bf16: Optional[bool] = field(
    default=False,
    metadata={"help": "Compute type of the model. If quantizing: this is also the compute type used for quantized computations. But since this is inference rather than training: the extra mantissa precision of float16 may be more useful than the exponent range of bfloat16."}
  )
  flash: Optional[bool] = field(
    default=False,
    metadata={"help": "Whether to replace the model code with togethercomputer's modeling_flash_llama.py, which uses Flash Attention 2 (via flash-attn) to accelerate model inference and reduce memory usage."}
  )

@dataclass
class MiscArguments:
  seed: Optional[int] = field(
    default=64,
    metadata={"help": "Random seed, for deterministic generation."}
  )
  compile: bool = field(
    default=False,
    metadata={"help": "Invoke torch.compile() on the model, with mode='max-autotune'. Requires PyTorch 2, CUDA, and either Python 3.10 or Python 3.11 with a recent torch nightly. Will make the first inference from the model take a bit longer, but subsequent inferences will be faster."}
  )
  system_prompt: str = field(
    default="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    metadata={"help": "The context which precedes the chat history. Can be used to influence the chatbot's responses."}
  )
  initial_input: Optional[str] = field(
    default=None,
    metadata={"help": "Initial message sent to the model. For example: What is $\sqrt{53}$ in simplest radical form?"}
  )
  # if you actually set the type hint to PromptStyle: you will find that HF/argparse assign a string anyway
  prompt_style: PromptStyleLiteral = field(
    default=PromptStyle.WizardCoderPython.value,
    metadata={"choices": [p.value for p in PromptStyle]}
  )
  chat_memory: bool = field(
    default=False,
    metadata={"help": "Whether chat sequence should accumulate a conversation context, or reset each time"}
  )
  reseed_each_prompt: bool = field(
    default=True,
    metadata={"help": "Reset seed before each user input"}
  )
  show_seed: bool = field(
    default=True,
    metadata={"help": "Show seed in prompt"}
  )
  measure_perf: bool = field(
    default=True,
    metadata={"help": "Print inference speed"}
  )

@dataclass
class GenerationArguments:
  # For more hyperparameters check:
  # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
  # Length arguments
  max_new_tokens: Optional[int] = field(
    default=2048,
    metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                      "if predict_with_generate is set."}
  )
  min_new_tokens : Optional[int] = field(
    default=None,
    metadata={"help": "Minimum number of new tokens to generate."}
  )

  # Generation strategy
  do_sample: Optional[bool] = field(default=False)
  num_beams: Optional[int] = field(default=1)
  num_beam_groups: Optional[int] = field(default=1)
  penalty_alpha: Optional[float] = field(default=None)
  use_cache: Optional[bool] = field(default=True)

  # Hyperparameters for logit manipulation
  temperature: Optional[float] = field(default=1.0)
  top_k: Optional[int] = field(default=50)
  top_p: Optional[float] = field(default=1.0)
  typical_p: Optional[float] = field(default=1.0)
  diversity_penalty: Optional[float] = field(default=0.0)
  repetition_penalty: Optional[float] = field(default=1.0)
  length_penalty: Optional[float] = field(default=1.0)
  no_repeat_ngram_size: Optional[int] = field(default=0)

def get_model(args: ModelArguments) -> LlamaForCausalLM:
  config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=args.trust_remote_code,
    cache_dir=args.cache_dir,
  )

  if args.flash and config.model_type == 'llama':
    updates: Dict[str, Union[str, int, float, bool, None]] = {}
    flash_model_name = 'Birchlabs/flash_llama--modeling_flash_llama.LlamaForCausalLM'
    if 'num_key_value_heads' not in config.__dict__:
      updates['num_key_value_heads'] = config.num_attention_heads
    if 'auto_map' in config.__dict__:
      if not ('AutoModelForCausalLM' in config.auto_map and 'flash' in config.auto_map['AutoModelForCausalLM']):
        updates['auto_map']['AutoModelForCausalLM'] = flash_model_name
    else:
      updates['auto_map'] = { 'AutoModelForCausalLM': flash_model_name }
    if 'rope_scaling' not in config.__dict__:
      updates['rope_scaling'] = { 'factor': (args.source_max_len + args.target_max_len)/config.max_position_embeddings, 'type': 'linear' }
    if 'pretraining_tp' not in config.__dict__:
      updates['pretraining_tp'] = 1
    if updates:
      config.update(updates)

  cuda_avail = torch.cuda.is_available()
  model_dtype = torch.bfloat16 if args.bf16 else torch.float16
  load_in_4bit = args.bits == 4 and cuda_avail
  load_in_8bit = args.bits == 8 and cuda_avail

  bnb_compute_dtype: torch.dtype = reify_dtype(args.bnb_compute_dtype)

  quantization_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=bnb_compute_dtype,
    bnb_4bit_use_double_quant=args.double_quant,
    bnb_4bit_quant_type=args.quant_type,
  ) if cuda_avail else None

  if not cuda_avail:
    logger.warning("You don't have CUDA, so we have turned off quantization. If you happen to be on a Mac: maybe you have enough unified memory to run in fp16 anyway…")
  
  model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    device_map='auto',
    quantization_config=quantization_config,
    torch_dtype=model_dtype,
    trust_remote_code=args.trust_remote_code,
    cache_dir=args.cache_dir,
  ).eval()
  model.config.torch_dtype=model_dtype

  return model

def main():
  hfparser = HfArgumentParser((ModelArguments, GenerationArguments, MiscArguments))
  model_args, generation_args, misc_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
  if extra_args:
    raise f"Received unsupported command-line args: {extra_args}"
  generation_config = GenerationConfig(**vars(generation_args))

  model: LlamaForCausalLM = get_model(model_args)

  set_seed(misc_args.seed)
  if misc_args.compile:
    torch.compile(model, mode='max-autotune')

  tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    # fast tokenizer required for WizardLM/WizardCoder-Python-34B-V1.0, because slow tokenizer doesn't come with added_tokens (required for {'[PAD]': 32000})
    use_fast=True,
    cache_dir=model_args.cache_dir,
  )
  generation_config.pad_token_id = tokenizer.pad_token_id

  stop_token_ids: List[int] = [tokenizer.eos_token_id]
  stop = StopOnTokens(stop_token_ids)
  stopping_criteria=StoppingCriteriaList([stop])

  history: List[Message] = [Message(Participant.System, misc_args.system_prompt)] if misc_args.system_prompt else []

  reset_ansi='\x1b[0m'
  cyan_ansi='\x1b[31;36m'
  blue_ansi='\x1b[31;34m'
  green_ansi='\x1b[31;32m'
  purple_ansi='\x1b[31;35m'

  participant_names: Dict[Participant, str] = {
    Participant.User: 'Instruction',
    Participant.Assistant: 'Response',
  }

  def alpaca_section(envelope: Message) -> str:
    participant, message = envelope
    if participant is Participant.System:
      return message
    return f'### {participant_names[participant]}:\n{message}'

  next_seed: Optional[int] = None

  first = True
  while True:
    seed: int = misc_args.seed if next_seed is None else next_seed
    if misc_args.reseed_each_prompt or first or next_seed is not None:
      set_seed(seed)

    try:
      prompt_ctx: str = f'[seed={seed}]' if misc_args.show_seed else ''
      if first and misc_args.initial_input is not None:
        user_input = misc_args.initial_input
        quote: str = f'{purple_ansi}{prompt_ctx}> '
        print(f'{quote}{user_input}')
      else:
        prompt: str = f'{purple_ansi}{prompt_ctx}$ '
        user_input = input(f'{blue_ansi}Type a message to begin the conversation…{reset_ansi}\n{prompt}' if first else prompt)
    except (KeyboardInterrupt, EOFError):
      sys.exit(0)
    print(reset_ansi, end='')

    first = False

    user_message: Message = Message(Participant.User, user_input)

    match misc_args.prompt_style:
      case PromptStyle.WizardCoderPython.value:
        # TODO
        chat_to_complete: str = '\n\n'.join([
          alpaca_section(message) for message in [
            *history,
            user_message,
            Message(Participant.Assistant, ''),
          ]
        ])
      case PromptStyle.CodeLlamaInstruct.value:
        # TODO
        chat_to_complete: str = user_input
      case PromptStyle.Bare.value:
        chat_to_complete: str = user_input
      case _:
        raise ValueError(f'Never heard of a {misc_args.prompt_style} PromptStyle.')

    tokenized_prompts: TokenizerOutput = tokenizer([chat_to_complete], return_tensors='pt', truncation=True)
    
    print(green_ansi, end='', flush=True)

    response = ''
    def on_text(message: str, stream_end = False):
      nonlocal response
      response += message
      print(message, end='', flush=True)

    streamer = CallbackTextIteratorStreamer(tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=True)

    try:
      inference_start: float = perf_counter()
      prediction: LongTensor = model.generate(
        input_ids=tokenized_prompts.input_ids.to(model.device),
        attention_mask=tokenized_prompts.attention_mask.to(model.device),
        generation_config=generation_config,
        do_sample=generation_config.temperature > 0.,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
      )
      # reset ANSI control sequence (plus line break)
      print(reset_ansi)
      # if you wanted to see the result, you can do so like this:
      # decode: List[str] = tokenizer.decode(prediction[0,tokenized_prompts.input_ids.size(-1):], skip_special_tokens=False, clean_up_tokenization_spaces=True)
      # print(decode)
      # pass
      # but we're already streaming it to the console via our callback
      inference_duration: float = perf_counter()-inference_start
      token_in_count: int = tokenized_prompts.input_ids.size(-1)
      token_out_count: int = prediction.size(-1) - token_in_count
      tokens_out_per_sec: float = token_out_count/inference_duration
      if misc_args.measure_perf:
        print(f'{cyan_ansi}ctx length: {token_in_count}\ntokens out: {token_out_count}\nduration: {inference_duration:.2f} secs\nspeed: {tokens_out_per_sec:.2f} tokens/sec{reset_ansi}')
    except (KeyboardInterrupt, SufficientResponse, EOFError):
      # reset ANSI control sequence (plus line break)
      print(reset_ansi)

    # we disable accumulation of conversation history by default, because WizardCoder is not advertised as being finetuned on multi-turn conversations,
    # but more importantly because I'd rather spend our 4k context length on a detailed answer for a single-turn than an incomplete answer for multiple turns.
    if misc_args.chat_memory:
      history += [
        user_message,
        Message(Participant.Assistant, response)
      ]

if __name__ == "__main__":
  main()