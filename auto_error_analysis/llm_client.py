"""
FROM REFPYDST

Methods for generating language model completions with the Codex family of models via the OpenAI API.

This file was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as originally
published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

@article{hu2022context,
  title={In-Context Learning for Few-Shot Dialogue State Tracking},
  author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2203.08568},
  year={2022}
}
"""
import logging
import os
from typing import List, TypedDict, Optional, Dict, Any, Callable

import openai
from openai import BadRequestError
from openai._exceptions import RateLimitError, APIError, APIConnectionError, OpenAIError

from refpydst.utils.general import check_argument
from refpydst.utils.speed_limit_timer import SpeedLimitTimer
from vllm import LLM, SamplingParams


TOO_MANY_TOKENS_FOR_ENGINE: str = "This model's maximum context length is"


class OpenAIAPIConfig(TypedDict):
    """
    A dictionary of config items for OpenAI API use
    """
    api_key: str
    organization: Optional[str]  # optional to toggle between a chosen one and API key default
    seconds_per_step: float


def _load_environment_codex_config() -> OpenAIAPIConfig:
    api_key: str = os.environ.get("OPENAI_API_KEY_JLAB_ORG") or os.environ.get("OPENAI_API_KEY")
    organization: str = os.environ.get("OPENAI_ORGANIZATION")
    check_argument(api_key, "must set an API key. Use environment variable OPENAI_API_KEY or otherwise provide "
                            "a CodexConfig")
    return {"api_key": api_key.strip(),  # easier than fixing a k8s secret
            "organization": organization,
            "seconds_per_step": 0.2}


class LLMClient():
    """
    Simplified client for working with Codex and OpenAI models, wraps openai client.
    """

    config: OpenAIAPIConfig
    engine: str
    stop_sequences: List[str]
    timer: SpeedLimitTimer

    def __init__(
        self, config: OpenAIAPIConfig = None, engine: str = "gpt-3.5-turbo-0125",
        stop_sequences: List[str] = None, quantization: str = None
    ) -> None:
        super().__init__()
        self.config = config or _load_environment_codex_config()
        self.engine = engine
        self.stop_sequences = stop_sequences or ['--', '\n', ';', '#']
        self.timer = SpeedLimitTimer(second_per_step=self.config['seconds_per_step'])  # openai limitation 20 query/min

        if 'llama' in engine.lower():
            self.model = LLM(model=self.engine, quantization=quantization, enforce_eager=True)
            self.tokenizer = self.model.get_tokenizer()
            self.terminators =  [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")    
            ]

    def greedy_lm_completion(self, prompt_text: str) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.
    
        :param prompt_text: prefix text for OpenAI Completion API
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            if 'llama' in self.engine.lower():
                samplig_params = SamplingParams(
                    n=1, best_of=1, max_tokens=120, 
                    temperature=0, stop=stop_sequences,
                    stop_token_ids=self.terminators
                )
                prompts = [
                    self.tokenizer.batch_decode(prompt, skip_special_tokens=False)[0] for prompt in prompt_text
                ]
                result = self.model.generate(prompts, sampling_params=samplig_params)
                if len(result) > 1:
                    completions = [{output.outputs[0].text: 1} for output in result]
                else:
                    completions = [{result[0].outputs[0].text: 1}]
        
                return completions
            
            else:
                args: Dict[str, Any] = {
                    "model": self.engine,
                    "messages": prompt_text,
                    "max_tokens": 120,
                    "logprobs": True,
                    "temperature": 0.0,
                    "stop": stop_sequences,
                }
                self.timer.step()
                result = openai.chat.completions.create(**args)
                completions = dict(zip(
                    [x.message.content for x in result.choices],
                    [sum(token.logprob for token in x.logprobs.content) for x in result.choices]
                ))
                return completions
            
        except BadRequestError as e:
            raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e
