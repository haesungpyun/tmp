import argparse
import copy
import datetime
import itertools
import json
import logging
import os
import random
import sys
import pprint
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple, Final, Type, Union
from tqdm import tqdm

import numpy as np
import wandb


from openai._exceptions import RateLimitError, APIError, OpenAIError
from refpydst.data_types import ExampleListDecoderConfig, CodexDecodingConfig, ExampleListDecoderType
from refpydst.data_types import Turn, CompletionParser, CodexPromptingRunConfig, MultiWOZDict

from refpydst.artifacts import output_dir_to_run_or_artifact_name
from refpydst.codex_client import CodexClient, PromptOverlengthError, LlamaClient
from refpydst.db.ontology import Ontology
from refpydst.generation_experiment import AbstractLMPromptingExperiment
from refpydst.prompting import PROMPT_VARIANTS, PromptGenerator, STOP_SEQUENCES, IC_DST
from refpydst.utils.general import read_json, subtract_dict, get_output_dir_full_path, WANDB_ENTITY, WANDB_PROJECT
from refpydst.utils.state_recorder import PreviousStateRecorder

from refpydst import evaluate_run_log
from refpydst.error_analysis import slot_level_f1, count_prompts_from_examples
from refpydst.evaluate_metrics import calc_prf, evaluate, compute_prf, compute_acc, calculate_token_f1
from refpydst.normalization.abstract_normalizer import AbstractNormalizer
from refpydst.retriever.abstract_example_retriever import ExampleRetriever
from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
import refpydst.prompt_formats.python.demo as python_demo
from refpydst.prompt_formats.python.completion_parser import parse_state_change
from refpydst.utils.dialogue_state import group_by_dial_id_and_turn
from refpydst.utils.dialogue_state import update_dialogue_state
from refpydst.utils.general import read_json, get_output_dir_full_path
from refpydst.utils.state_recorder import PreviousStateRecorder
from refpydst.wandb_step_logger import WandbStepLogger

_DECODER_CLASSES: Dict[ExampleListDecoderType, Type[AbstractExampleListDecoder]]


class CodexExperiment(AbstractLMPromptingExperiment):
    """
    Class for managing an experiment with Codex
    """

    test_set: List[Turn]
    use_gold: bool
    prompt_format: str
    only_use_turn_idx: int
    prediction_recorder: PreviousStateRecorder
    prompt_generator: PromptGenerator
    retriever: ExampleRetriever
    num_examples: int
    ontology_file_path: str
    demonstration_mapping: Optional[Dict[str, Dict[str, List[Turn]]]]
    train_set: List[Turn]
    completion_parser: CompletionParser
    mwz_ver: str
    demonstration_decoder: AbstractExampleListDecoder
    output_dir: str
    ontology: Ontology
    normalizer: AbstractNormalizer
    format_example: Optional[Turn]
    logger: WandbStepLogger
    
    ontology: Ontology
    codex_client: CodexClient
    num_distinct_demonstrations: Optional[int]
    lm_decoding_config: CodexDecodingConfig
    min_null_sequence_log_probability: float
    min_null_token_log_probability: float

    def __init__(self, test_set_path: str, ontology_file_path: str, use_gold: bool = False,
                 prompt_format: str = None, turn: int = -1, num_examples: int = 10, retriever_dir: str = None,
                 train_set_path: str = None, demonstration_mapping_path: str = None,
                 codex_engine: str = "gpt-3.5-turbo", mwz_ver: str = "2.4",
                 retriever_type: str = None, decoder_config: Optional[ExampleListDecoderConfig] = None,
                 lm_decoding_config: Optional[CodexDecodingConfig] = None,
                 format_example: Optional[Turn] = None,
                 **kwargs) -> None:
        super().__init__(
            test_set_path=test_set_path,    # mw24_100p_test.json
            ontology_file_path=ontology_file_path,  # db/multiwoz/2.4/ontology.json
            use_gold=use_gold,  # False
            prompt_format=prompt_format,    # python-prompt
            turn=turn,  # -1
            num_examples=num_examples,  # 10
            format_example=format_example,  # None
            retriever_dir=retriever_dir,    # runs/retriever/mw21_1p_train/referred_states/split_v2/
            train_set_path=train_set_path,  # mw21_1p_train_v2.json
            demonstration_mapping_path=demonstration_mapping_path,  # None
            mwz_ver=mwz_ver,    # 2.4
            retriever_type=retriever_type,  # EmbeddingRetriever
            decoder_config=decoder_config,  # {'decoder_type': 'max_emb_distance', 'discount_factor': 0.2, 'from_n_possible': 100}
            **kwargs   
        )
        self.lm_decoding_config = lm_decoding_config
        self.beam_search_config = None
        if self.lm_decoding_config is not None:
            self.beam_search_config = {'beam_size': lm_decoding_config.pop('beam_size')} if lm_decoding_config.get('beam_size') else None
        
        min_null_token_prob: float = self.lm_decoding_config and self.lm_decoding_config.get('min_token_null_probability', 0) or 0
        self.min_null_token_log_probability = np.log(min_null_token_prob) if min_null_token_prob != 0 else sys.float_info.min
        min_null_sequence_prob: float = self.lm_decoding_config and self.lm_decoding_config.get('min_null_probability', 0) or 0
        self.min_null_sequence_log_probability = np.log(min_null_sequence_prob) if min_null_sequence_prob != 0 else sys.float_info.min
        if codex_engine.startswith('gpt'):
            self.codex_client = CodexClient(engine=codex_engine, stop_sequences=STOP_SEQUENCES.get(self.prompt_format))
        elif codex_engine.startswith('llama') or codex_engine.startswith('meta') or 'llama' in codex_engine.lower():
            # self.codex_client = LlamaClient(engine=codex_engine, stop_sequences=STOP_SEQUENCES.get(self.prompt_format), quantization=kwargs.get('quantization'))
            self.codex_client = LlamaClient(engine=codex_engine, stop_sequences=STOP_SEQUENCES.get(self.prompt_format), quantization=kwargs.get('quantization'), beam_search_config=self.beam_search_config)
        
        self.num_sampling_iteration = kwargs.get("num_sampling_iteration", 5)
        self.num_samples = kwargs.get("num_samples", 10)
        self.num_final_samples = kwargs.get("num_final_shots", self.num_samples)
        self.score_type = kwargs.get("score_type", "score_delta")
        self.add_guidelines = kwargs.get("add_guidelines", True)


    def generate_completion(self, prompt_text: str, data_item: Turn, examples: List[Turn], just_return_completion:bool = False) -> Tuple[
        Dict[str, float], List[Turn]]:
        # codex completion
        complete_flag = False
        parse_error_count = 0
        other_error_cnt = 0
        completions: Dict[str, float] = {}
        while not complete_flag and parse_error_count < 5 and other_error_cnt < 5:
            try:
                if self.lm_decoding_config is None or self.lm_decoding_config.get("method", "greedy") in ["greedy", "beam_search"]:
                    completions = self.codex_client.greedy_lm_completion(prompt_text)
                
            except PromptOverlengthError as e:
                logging.warning(e)
                logging.info("prompt overlength, retrying with fewer examples")
                examples = examples[1:]
                prompt_text = self.get_prompt_text_dict(data_item=data_item, examples=examples)
                other_error_cnt += 1
            except ValueError as e:
                logging.exception(e)
                other_error_cnt += 1
                raise e
            except (RateLimitError, APIError, OpenAIError) as e:
                # client will manage sleeping/timing
                logging.exception(e)
                other_error_cnt += 1
            except BaseException as e:
                logging.exception(e)
                if type(e) == KeyboardInterrupt:
                    raise e
                other_error_cnt += 1
            else:
                # interesting python idiom: try/except/else: else executes if there is no exception of any typ e in the
                # try block (kind of un-needed here, but can be used with finally to be try/except/else/finally)
                try:
                    # check if CODEX is crazy
                    predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
                    # for now, just verify our best completion is parse-able
                    tmp_comp = list(completions[0].keys())[0]
                    temp_parse = self.completion_parser(tmp_comp, predicted_context)
                    complete_flag = True
                except Exception:
                    parse_error_count += 1

        if not complete_flag:
            if just_return_completion:
                return completions, examples
            raise ValueError("unable to generate completion")
        
        return completions, examples


    def make_prediction(self, data_item, train_by_dial_id, prompt_text_dict, examples, save_prediction=False):
        batch_predicted_slot_values: MultiWOZDict = []
        batch_pred_turn_slot_values: MultiWOZDict = []
        predicted_prior_context: MultiWOZDict = None
        completions = None  # the except block will print it, which can be confusing if its from the previous turn
        try:
            dial_id, turn_id = data_item['ID'], data_item['turn_id']         
            
            completions, examples = self.generate_completion(prompt_text_dict, data_item, examples)
            best_completion = [comp.strip().replace('agent.state.', '') for dic in completions for comp,_ in dic.items()]

            # aggregate the prediction and the history states
            predicted_prior_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
            batch_predicted_slot_values = [self.completion_parser(comp, predicted_prior_context) for comp in best_completion]
            batch_pred_turn_slot_values = [self.completion_parser(comp, predicted_prior_context) for comp in best_completion]

        except Exception as e:
            best_completion = "error"
            completions, examples = self.generate_completion(prompt_text_dict, data_item, examples, just_return_completion=True)
            print(f"the output could not be parsed successfully: {completions}", e)
            data_item['not_valid'] = 1
            data_item['completion'] = completions
        
        batch_predicted_slot_values = [self.normalizer.normalize(s_v) for s_v in batch_predicted_slot_values]
        batch_pred_turn_slot_values = [self.normalizer.normalize(s_v) for s_v in batch_pred_turn_slot_values]
        
        # merge context and prediction
        if self.use_gold:
            prior_dialogue_state = data_item['last_slot_values'].copy()
        else:
            prior_dialogue_state = self.prediction_recorder.retrieve_previous_turn_state(data_item).copy()

        batch_all_slot_values = [update_dialogue_state(prior_dialogue_state, predicted_slot_values) for predicted_slot_values in batch_predicted_slot_values]
        # some slots may contain multiple values
        batch_all_slot_values = [{k: v.split('|')[0] for k, v in all_slot_values.items()} for all_slot_values in batch_all_slot_values]

        # record current turn prediction
        if save_prediction:
            self.prediction_recorder.add_state(data_item, batch_all_slot_values[0])

        return batch_all_slot_values, batch_predicted_slot_values, predicted_prior_context, batch_pred_turn_slot_values,  best_completion, completions        


    def compute_jga(self, prediction: MultiWOZDict, gold: MultiWOZDict):   
        for key in gold.keys():
            # if the gold value supports multiple ground truth values, and we predicted one, set the single-gold value to
            # the one we predicted.
            if '|' in gold[key]:
                gold_values = gold[key].split('|')
                if key in prediction and prediction[key] in gold_values:
                    gold[key] = prediction[key]

        # joint-goal can be computed with dict match
        return 1 if prediction == gold else 0

    def label_id(self, data_item: Turn) -> str:
        return data_item['ID'] + "_turn_" + str(data_item['turn_id'])

    def run(self) -> Tuple[List[Turn], Dict[str, Any]]:
        jga_by_turn_id = defaultdict(list)  # use to record the accuracy

        selected_set: List[Turn] = self.test_set
        # if needed, only evaluate on particular turns (analysis purpose)
        if self.only_use_turn_idx >= 0:
            if not self.use_gold:
                raise ValueError("can only evaluate particular turn when using gold context")
            selected_set = [d for d in self.test_set if len(d['dialog']['usr']) == self.only_use_turn_idx + 1]

        # start experiment
        running_log: List[Turn] = []
        n_total: int = 0
        n_correct: int = 0
        total_acc: float = 0
        total_f1: float = 0
        train_by_dial_id = group_by_dial_id_and_turn(self.train_set)

        for data_item_idx, data_item in tqdm(enumerate(selected_set)):
            n_total += 1
            if 'pred' in data_item:
                if isinstance(data_item['pred'], list):
                    data_item['pred'] = data_item['pred'][0]
                self.prediction_recorder.add_state(data_item, data_item['pred'])
                running_log.append(data_item)
                this_jga, this_acc, this_f1 = evaluate(data_item['pred'], data_item['slot_values'])
                total_acc += this_acc
                total_f1 += this_f1

                if this_jga:
                    n_correct += 1
                    jga_by_turn_id[data_item['turn_id']].append(1)
                    print("\n=====================correct!=======================")
                else:
                    jga_by_turn_id[data_item['turn_id']].append(0)
                    print("\n=====================wrong!=======================")
                self.logger.log({"current_jga": n_correct / n_total, "n_total": n_total})
                self.logger.step()
                print("\n")

                if data_item_idx > 20:
                    with open(os.path.join(self.output_dir, "running_log.json"), 'w') as f:
                        json.dump(running_log, f)
                continue

            retrieved_examples: List[Turn] = []
            if self.use_gold:
                retrieved_examples.extend(self.retriever.item_to_best_examples(data_item, k=self.num_examples,
                                                                decoder=self.demonstration_decoder))
            else:
                # we have remaining examples to retriever (in most few-shot settings, all of them)
                predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
                modified_item = copy.deepcopy(data_item)
                modified_item['last_slot_values'] = predicted_context
               
                retrieved_examples = self.retriever.item_to_best_examples(
                    modified_item, k=self.num_examples, decoder=self.demonstration_decoder)[::-1]
            
            retrieved_example_ids = [self.label_id(e) for e in retrieved_examples]
            assert len(set(retrieved_example_ids)) == len(retrieved_example_ids), f"The {data_item_idx}-th data {data_item['ID']}, {data_item['turn_id']}. The retrieved examples are not unique. set: {len(set(retrieved_example_ids))}, total: {len(retrieved_example_ids)}"

            sampling_pool = []
            for idx in range(self.num_sampling_iteration):
                random.seed(idx)
                sampling_pool += random.sample(retrieved_examples, len(retrieved_examples))

            num_sub_group = len(retrieved_examples)//self.num_samples
            data_item['sampling_exp'] = {}
            data_item['sampling_exp']['exp'] = []
            data_item['sampling_exp']['scores'] = []

            sample_idx = 0
            
            for iteration in range(self.num_sampling_iteration):
                prompt_list = []
                iter_log = {f"iter_{iteration}": []}
                
                iter_scores = {}
                for key in ['occurence', 'score_delta', 'score_full', 'influence_delta', 'influence_full']:
                    iter_scores[key] = {ids: 0 for ids in retrieved_example_ids}

                examples = {}
                example_ids = {}
                for step, _ in enumerate(range(0, len(retrieved_examples), self.num_samples)):
                    # sample the examples to make prompt and make prediction
                    examples[step] = sampling_pool[sample_idx : sample_idx + self.num_samples] 
                    sample_idx += self.num_samples   
                    example_ids[step] = [self.label_id(x) for x in examples[step]]
                    
                    prompt_text_dict: Final[str] = self.get_prompt_text_dict(data_item, examples[step], zero_one_shot=False)
                    prompt_token_ids = self.codex_client.tokenizer.apply_chat_template(
                        prompt_text_dict[f'{len(examples[step])}-shot'], add_generation_prompt=True, return_tensors="pt")
                    prompt_list.append(prompt_token_ids)  
                # import pdb
                # pdb.set_trace()
                batch_all_slot_values, batch_predicted_slot_values, predicted_prior_context, \
                batch_pred_turn_slot_values,  best_completion, completions = \
                    self.make_prediction(data_item, train_by_dial_id, prompt_list, examples, save_prediction=False)
                
                # record the scores
                batch_delta_jga = [self.compute_jga(pred_turn_slot_values, data_item['turn_slot_values']) for pred_turn_slot_values in batch_pred_turn_slot_values]
                batch_full_jga = [self.compute_jga(all_slot_values, data_item['slot_values']) for all_slot_values in batch_all_slot_values]
                for idx, ex_id_list in example_ids.items():
                    for ex_id in ex_id_list:
                        iter_scores['occurence'][ex_id] += 1
                        for key in ['score_delta', 'score_full', 'influence_delta', 'influence_full']:
                            iter_scores[key][ex_id] += batch_delta_jga[idx] if 'delta' in key else batch_full_jga[idx]

                    for neg_ex_id in set(retrieved_example_ids) - set(ex_id_list):
                        iter_scores['influence_delta'][neg_ex_id] -= (1/(num_sub_group-1))*batch_delta_jga[idx]
                        iter_scores['influence_full'][neg_ex_id] -= (1/(num_sub_group-1))*batch_full_jga[idx]
                
                # record the predictions
                for idx in range(len(batch_all_slot_values)):
                    step_log= {}
                    step_log['pred'] = batch_all_slot_values[idx]
                    step_log['pred_delta_slot_values'] = batch_predicted_slot_values[idx]
                    step_log['pred_prior_context'] = predicted_prior_context or {}
                    step_log['pred_turn_slot_values'] = batch_pred_turn_slot_values[idx]
                    step_log['completion'] = best_completion[idx]
                
                    step_log['prompt_counts'] = count_prompts_from_examples(examples[idx])
                    step_log['examples'] = [(e['ID'], e['turn_id']) for e in examples[idx]]

                    iter_log[f'iter_{iteration}'].append(step_log)

                print(f"\n======= iteration: {iteration+1} / {self.num_sampling_iteration} =======")
                print(f"few-shot best completions: {best_completion}")
                print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
                print(f"system response: {data_item['dialog']['sys'][-1]}")
                print(f"user response: {data_item['dialog']['usr'][-1]}")
                print(f"gold turn change: {pprint.pformat(data_item['turn_slot_values'])}")
                print(f"pred turn change: {pprint.pformat(batch_pred_turn_slot_values)}")
                print()
                print(f"gold states: {pprint.pformat(data_item['slot_values'])}")
                print(f"pred states: {pprint.pformat(batch_all_slot_values)}")
                print(f"Delta JGA: {batch_delta_jga}")
                print(f"Full JGA: {batch_full_jga}")

                data_item['sampling_exp']['exp'].append(iter_log)
                data_item['sampling_exp']['scores'].append(iter_scores)     
            data_item['final_scores'] = {}
            # Aggregate the scores in each iteration to final_score
            for score_idx, scores in enumerate(data_item['sampling_exp']['scores']):
                for key in scores:
                    if key not in data_item['final_scores']:
                        data_item['final_scores'][key] = copy.deepcopy(scores[key])
                    else:
                        for ex_id in retrieved_example_ids:
                            data_item['final_scores'][key][ex_id] += scores[key][ex_id]

            best_ex_id_score = sorted(data_item['final_scores'][self.score_type].items(), key=lambda x: x[1], reverse=True)[:self.num_final_samples][::-1]
            examples = []
            for (ex_id, score) in best_ex_id_score:
                    for data in retrieved_examples:
                        if self.label_id(data) == ex_id:
                            examples.append(data)
                            break

            data_item['best_example'] = examples
            prompt_text_dict: Final[str] = self.get_prompt_text_dict(data_item, examples, zero_one_shot=False)
            data_item['prompt'] = prompt_text_dict

            prompt_token_ids = self.codex_client.tokenizer.apply_chat_template(
                prompt_text_dict[f'{len(examples)}-shot'], add_generation_prompt=True, return_tensors="pt")

            all_slot_values, predicted_slot_values, predicted_prior_context, pred_turn_slot_values, best_completion, completions = \
                self.make_prediction(data_item, train_by_dial_id, [prompt_token_ids], examples, save_prediction=True)

            # record the predictions
            data_item['pred'] = all_slot_values
            data_item['pred_delta_slot_values'] = predicted_slot_values
            data_item['pred_prior_context'] = predicted_prior_context or {}
            data_item['pred_turn_slot_values'] = pred_turn_slot_values
            data_item['completion'] = best_completion
            data_item['all_completions'] = completions
            data_item['num_solutions'] = len(completions)
            data_item['prompt_counts'] = count_prompts_from_examples(examples)
            data_item['examples'] = [(e['ID'], e['turn_id']) for e in retrieved_examples]
            running_log.append(data_item)
            
            print(f"\nfew-shot completions: {completions}")
            print(f"few-shot best completion: {best_completion}")
            print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
            print(f"system response: {data_item['dialog']['sys'][-1]}")
            print(f"user response: {data_item['dialog']['usr'][-1]}")
            print(f"gold turn change: {pprint.pformat(data_item['turn_slot_values'])}")
            print(f"pred turn change: {pprint.pformat(pred_turn_slot_values)}")
            print(f"gold states: {pprint.pformat(data_item['slot_values'])}")
            print(f"pred states: {pprint.pformat(all_slot_values)}")

            this_jga, this_acc, this_f1 = evaluate(all_slot_values[0], data_item['slot_values'])
            total_acc += this_acc
            total_f1 += this_f1

            if this_jga:
                n_correct += 1
                jga_by_turn_id[data_item['turn_id']].append(1)
                print("\n=====================correct!=======================")
            else:
                jga_by_turn_id[data_item['turn_id']].append(0)
                print("\n=====================wrong!=======================")
            print({"current_jga": n_correct / n_total, "n_correct": n_correct, "n_total": n_total})
            self.logger.log({"current_jga": n_correct / n_total,"n_correct": n_correct, "n_total": n_total})
            self.logger.step()
            print("\n")

            # write out running log regularly, in-case we stop a run. Give some buffer in-case we accidentally start,
            # and didn't want to over-write
            if data_item_idx > 20:
                with open(os.path.join(self.output_dir, "running_log.json"), 'w') as f:
                    json.dump(running_log, f, indent=2)
        stats = evaluate_run_log.evaluate_logs(running_log, test_set=self.test_set)
        slot_prf = slot_level_f1(running_log, tp_means_correct=True)
        self.logger.log({f"f1/{slot_name}": f1 for slot_name, (_, f1) in slot_prf.items()})
        self.logger.log({f"precision/{slot_name}": calc_prf(f1_dict).precision for slot_name, (f1_dict, f1) in slot_prf.items()})
        self.logger.log({f"recall/{slot_name}": calc_prf(f1_dict).recall for slot_name, (f1_dict, f1) in slot_prf.items()})

        turn_acc_table: wandb.Table = wandb.Table(data=[
            [f"{turn_id}", acc] for turn_id, acc in stats['turn_accuracies'].items()
        ], columns=['turn_id', 'accuracy'])
        stats['turn_accuracies'] = wandb.plot.bar(turn_acc_table, "turn_id", "accuracy", title="accuracy by turn id")
        self.logger.log(stats)

        # get per-domain stats
        by_domain_stats: Dict[str, Dict[str, Any]] = evaluate_run_log.evaluate_on_domains(running_log, self.test_set)
        flattened_domain_stats: Dict[str, Any] = {}
        for domain, domain_scores in by_domain_stats.items():
            for metric, value in domain_scores.items():
                flattened_domain_stats[f"{domain}-{metric}"] = value
        self.logger.log(flattened_domain_stats)

        # log running_log as an artifact
        self.logger.step()
        return running_log, stats
   

def main(train_fn: str, retriever_dir: str, output_dir: str, test_fn: str, prompt_format: str = IC_DST,
         mwz_ver: str = "2.4",
         codex_engine: str = "gpt-3.5-turbo", demonstration_mapping_path: str = None,
         retriever_type: str = "EmbeddingRetriever", decoder_config: ExampleListDecoderConfig = None,
         lm_decoding_config: Optional[CodexDecodingConfig] = None,
         artifact_cache: str = None,
         format_example: Optional[Turn] = None, num_examples: int = 10, **kwargs) -> None:
    # create the output folder
    if os.path.exists(output_dir):
        output_dir = output_dir + "_" + str(datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9))).strftime('%m%d_%H%M'))
    os.makedirs(output_dir, exist_ok=True)
    # write out this experiment's configuration
    exp_config: Dict[str, Union[str, int]] = dict(locals())
    with open(os.path.join(output_dir, "exp_config.json"), 'w') as f:
        json.dump(exp_config, f, indent=4)

    # read the ontology and the test set
    ontology_file_path = f"db/multiwoz/{mwz_ver}/ontology.json"
    if mwz_ver == '2.1':
        test_set_path = test_fn or "./data/mw21_100p_test.json"
    else:
        test_set_path = test_fn or "./data/mw24_100p_test.json"

    experiment: CodexExperiment = CodexExperiment(
        artifact_cache=artifact_cache,  # None
        train_set_path=train_fn,    # mw21_1p_train_v2.json
        retriever_dir=retriever_dir,    # runs/retriever/mw21_1p_train/referred_states/split_v2/
        test_set_path=test_set_path,    # mw24_100p_test.json
        ontology_file_path=ontology_file_path,  # db/multiwoz/2.4/ontology.json
        num_examples=num_examples,  # 10
        prompt_format=prompt_format,    # python-prompt
        demonstration_mapping_path=demonstration_mapping_path,  # None
        codex_engine=codex_engine,  # gpt-3.5-turbo-instruct
        mwz_ver=mwz_ver,    # 2.4
        retriever_type=retriever_type,  # EmbeddingRetriever
        decoder_config=decoder_config,  # {'decoder_type': 'max_emb_distance', 'discount_factor': 0.2, 'from_n_possible': 100}
        lm_decoding_config=lm_decoding_config,  # {'method': 'top_p', 'top_p': 0.9, 'stop_token': ';', 'max_mi_candidates': 100, 'null_prompt_format': 'python-prompt', 'null_prompt_weight': 1.0, 'min_null_probability': 0.0, 'min_token_null_probability': 0.0}
        output_dir=output_dir,  # /home/haesungpyun/RefPyDST/outputs/runs/codex/mw21_1p_train/python/top_p_0_9_x_max_emb_02_canonical_beta_0_4/split_v2
        format_example=format_example,  # None
        **kwargs    # {'retriever_args': {'state_transformation': 'ref_aware'}, 'run_name': 'runs-codex-mw21_1p_train-python-top_p_0_9_x_max_emb_02_canonical_beta_0_4-split_v2'}
    )

    try:
        running_log, stats = experiment.run()
    finally:
        artifact: wandb.Artifact = wandb.Artifact(wandb.run.name, type="run_output")
        artifact.add_dir(experiment.output_dir)
        wandb.log_artifact(artifact)

    # write out running log
    with open(os.path.join(output_dir, "running_log.json"), 'w') as f:
        json.dump(running_log, f)

    if len(running_log) == len(experiment.test_set):
        run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
        run.tags.append("complete_run")
        run.update()

if __name__ == "__main__":
    os.environ['REFPYDST_DATA_DIR'] = "..../data"
    os.environ['REFPYDST_OUTPUTS_DIR'] = "..../outputs"

    import warnings
    warnings.warn("This script is deprecated. Please use the `run_codex_experiment.py` script instead.")

    if os.path.exists(sys.argv[1]):
        run_file: str = sys.argv[1]
        # arguments are input from a configuration file if the first argument to the program is a valid file
        args: CodexPromptingRunConfig = read_json(run_file)
        if 'output_dir' not in args:
            args['output_dir'] = get_output_dir_full_path(run_file.replace('.json', ''))
        if not 'run_name' in args:
            args['run_name'] = output_dir_to_run_or_artifact_name(args['output_dir'])
    else:
        # otherwise, try to parse from argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_fn', type=str, help="training data file (few-shot or full shot)",
                            required=True)  # e.g. "./data/mw21_10p_train_v3.json"
        parser.add_argument('--prompt_format', type=str, choices=PROMPT_VARIANTS,
                            help=f"prompt format variant, among: {', '.join(PROMPT_VARIANTS)}",
                            default="IC-DST")  # e.g. "IC-DST"
        parser.add_argument('--retriever_dir', type=str, required=True,
                            help="sentence transformer saved path")  # "./retriever/expts/mw21_10p_v3_0304_400_20"
        parser.add_argument('--output_dir', type=str, default="./expts/debug",
                            help="directory to save running log and configs")
        parser.add_argument('--mwz_ver', type=str, default="2.1", choices=['2.1', '2.4'], help="version of MultiWOZ")
        parser.add_argument('--codex_engine', type=str, default="gpt-3.5-turbo", choices=["text-davinci-002"],
                            help="version of GPT-3/Codex to complete with")
        parser.add_argument('--demonstration_mapping_path', type=str, default=None,
                            help="if provided, don't use retriever to find nearby dialogue turns, and instead use those "
                                 "provided in the mapping load-able at this path. It should contain a dictionary of the"
                                 "form: {dial_id: {turn_id: [(dial_id, turn_id), ...]}, ...}")
        parser.add_argument('--test_fn', type=str, default='', help="file to evaluate on, empty means use the test set")
        parser.add_argument('--retriever_type', type=str, default='EmbeddingRetriever',
                            help="what kind of retriever to use")
        args = parser.parse_args()
        args = vars(args)

    default_run_name: str = output_dir_to_run_or_artifact_name(args['output_dir'])
    default_run_group: str = default_run_name.rsplit('-', maxsplit=1)[0]
    wandb_entity: str = os.environ.get(WANDB_ENTITY, "haesung-pyun-seoul-national-university")
    wandb_project: str = os.environ.get(WANDB_PROJECT, "error_TOD")
    run = wandb.init(config=args, project=wandb_project, entity=wandb_entity,
                     name=args.get("run_name", default_run_name), notes=args.get("run_notes", None),
                     group=args.get("run_group", default_run_group),
                     tags=args.get("run_tags", None))
    main(**args)