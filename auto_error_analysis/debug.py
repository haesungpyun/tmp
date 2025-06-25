import abc
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Literal
import copy


from evaluate_run_log import evaluate_logs
from evaluate_metrics import evaluate, slot_level_f1
from normalization.data_ontology_normalizer import DataOntologyNormalizer
from db.ontology import Ontology
from data_types import SlotName, SlotValue, MultiWOZDict
from utils import validate_path_and_make_abs_path,read_json, save_analyzed_log, load_analyzed_log
from bs_utils import compute_dict_difference, sort_data_item, unroll_or, update_dialogue_state

from completion_parser import PARSING_FUNCTIONS

import torch
from vllm import LLM, SamplingParams


class AbstractAnalyzer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def categorize_error_case(self, pred_bs, gold_bs) -> Dict[SlotName, SlotValue]:
        return NotImplementedError('')

    @abc.abstractmethod
    def analyze(self, pred_bs, gold_bs) -> Dict[SlotName, SlotValue]:
        return NotImplementedError('')

class ErrorAnalyzer(AbstractAnalyzer):
    def __init__(
            self,
            train_data_path: str = None,
            result_file_path: str = None, 
            output_dir_path: str = './',
            ontology_path: str = './src/refpydst/db/multiwoz/2.4/ontology.json',
            parsing_func: str = 'naive_parse_nl_completion',
            special_values: List[str] = ['dontcare', '[DELETE]'],
            use_llm: bool = False,
            llm_config: Dict[str, Any] = {},
    ):  
        """
        Initializes the error analyzer.
            
        Args:
            train_data_path (str): The path to the training data. This is used to get the normalizer.
            result_file_path (str): The path to the result file. 
            output_dir_path (str): The path to save the analyzed log.
            ontology_path (str): The path to the ontology file.
            parsing_func (function): The parsing function to use for iterative parsing.
                                     [naive_parse_nl_completion(default), naive_iterative_parsing, parse_python_modified, parse_state_change, parse_python_completion]
            special_values (List[str]): The special values to consider as errors.
        """
        
        train_data = read_json(train_data_path)
        self.normalizer = self.get_normalizer(train_data, ontology_path)

        self.result_file_path = validate_path_and_make_abs_path(result_file_path)
        self.output_dir_path = output_dir_path or result_file_path
        self.output_dir_path = validate_path_and_make_abs_path(self.output_dir_path, is_output_dir=True)

        try:
            self.parsing_func = PARSING_FUNCTIONS[parsing_func]
        except KeyError:
            raise ValueError(f"Invalid parsing function: {parsing_func}. Choose from {list(PARSING_FUNCTIONS.keys())}")
        
        self.special_values = special_values or ['dontcare', '[DELETE]']

        self.use_llm = use_llm
        
        if self.use_llm:     
            if not llm_config:
                llm_config = {
                    "engine":"../models/Meta-Llama-3-70B-Instruct-GPTQ",
                    "quantization":"GPTQ",
                }
                
            self.LLM = LLM(model=llm_config['engine'], quantization=llm_config['quantization'], enforce_eager=True)
            self.tokenizer = self.LLM.get_tokenizer()
            self.terminators =  [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")    
            ]
    
            self.sampling_params = SamplingParams(
                n=1, best_of=1, max_tokens=120, 
                temperature=0.2, stop_token_ids=self.terminators)
             
    
    def get_normalizer(self, train_data, ontology_path):
        
        ontology_path = validate_path_and_make_abs_path(ontology_path)
        return DataOntologyNormalizer(
                Ontology.create_ontology(),
                # count labels from the train set
                supervised_set=train_data,
                # make use of existing surface form knowledge encoded in ontology.json, released with each dataset
                # see README.json within https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip
                counts_from_ontology_file=ontology_path
        )

    def record_error_and_update_visited(
        self,
        error_dict: Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]],
        error_name: str,
        error_s_v_pairs: Union[Tuple[SlotName, SlotValue], Tuple[SlotName, SlotValue, SlotName, SlotValue]],
        visited_pairs: List[Tuple[SlotName, SlotValue]] = []
    ) -> Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]:
        """
        Records an error in the error dictionary and updates the list of visited pairs.

        Args:
            error_dict (Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]]): The error dictionary.
            error_name (str): The name of the error.
            error_s_v_pairs (Union[Tuple[SlotName, SlotValue], Tuple[SlotName, SlotValue, SlotName, SlotValue]]): The slot and value pair(s) associated with the error.
            visited_pairs (List[Tuple[SlotName, SlotValue]]): List of visited slot and value pairs

        Returns:
            Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]: Updated error dictionary and visited pairs list.
        """
        # Append the error to the error dictionary
        if error_name is None or error_s_v_pairs is None:
            return error_dict, visited_pairs
        if (error_name, error_s_v_pairs) in error_dict.get('error', []):
            return error_dict, visited_pairs
    
        error_dict.setdefault('error', []).append((error_name, error_s_v_pairs))
        if visited_pairs is None:
            return error_dict, visited_pairs
        
        # Update the visited pairs list
        assert len(error_s_v_pairs) in [2, 4]

        if ('error_prop' in error_name):
            if 'hall' in error_name:
                visited_pairs.append((error_s_v_pairs[-2], error_s_v_pairs[-1]))
            elif 'miss' in error_name:
                visited_pairs.append((error_s_v_pairs[0], error_s_v_pairs[1]))
        else:
            visited_pairs.append((error_s_v_pairs[0], error_s_v_pairs[1]))
            if len(error_s_v_pairs) == 4:
                visited_pairs.append((error_s_v_pairs[2], error_s_v_pairs[3]))

        return error_dict, visited_pairs

    def preprocess_belief_state(
        self,
        analyzed_item: dict,
        prev_item: dict,
    ):
        # get the previous item's dialogue state and the previous predicted dialogue state
        prev_gold_bs, prev_pred_bs = unroll_or(
            gold=analyzed_item['last_slot_values'], pred=prev_item.get(f'pred_{self.parsing_func.__name__}', {})
        )
        analyzed_item[f'last_pred_{self.parsing_func.__name__}'] = prev_pred_bs

        # Delta(State Change) Belief State parsing. Parse the completion and normalize considering surface forms
        pred_delta_bs = analyzed_item.get(f'pred_delta_{self.parsing_func.__name__}', None)
        if pred_delta_bs is None:
            return_tuple = self.parsing_func(analyzed_item['completion'], state=prev_pred_bs)
            if isinstance(return_tuple, tuple) and len(return_tuple) == 2:
                parsed_pred_delta_bs, error_reason = return_tuple
            else:
                parsed_pred_delta_bs, error_reason = return_tuple, None
            if parsed_pred_delta_bs:
                parsed_pred_delta_bs = self.normalizer.normalize(raw_parse=parsed_pred_delta_bs) if 'DELETE' not in str(parsed_pred_delta_bs) else parsed_pred_delta_bs
            analyzed_item[f'pred_delta_{self.parsing_func.__name__}'] = parsed_pred_delta_bs
        gold_delta_bs, parsed_pred_delta_bs = unroll_or(gold=analyzed_item['turn_slot_values'], pred=parsed_pred_delta_bs)
        
        # Accumulated Dialogue State Belief State. Update the pred with parsed delta(State Change) belief state
        tmp_pred_delta = copy.deepcopy(parsed_pred_delta_bs) if parsed_pred_delta_bs else {}
        pred_bs = update_dialogue_state(context=prev_pred_bs, normalized_turn_parse=tmp_pred_delta) 
        gold_bs, pred_bs = unroll_or(gold=analyzed_item['slot_values'], pred=pred_bs)
        analyzed_item[f'pred_{self.parsing_func.__name__}'] = pred_bs

        return gold_bs, pred_bs, gold_delta_bs, parsed_pred_delta_bs, prev_gold_bs, prev_pred_bs, error_reason
    
    
    def detect_delta_missings(
        self, 
        delta_miss_gold: MultiWOZDict,  
        delta_over_pred: MultiWOZDict, 
        gold_delta_bs: MultiWOZDict, 
        pred_delta_bs: MultiWOZDict,
        prev_pred_bs: MultiWOZDict,
        visited: List[Tuple[SlotName, SlotValue]]
    )-> Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]:
        """
        Detects missing values in the prediction compared to the gold standard and records errors.

        Args:
            delta_miss_gold (MultiWOZDict): Missing values in the gold standard.
            delta_over_pred (MultiWOZDict): Over-predicted values.
            gold_delta_bs (MultiWOZDict): The gold standard delta belief state.
            pred_delta_bs (MultiWOZDict): The predicted delta belief state.
            visited (List[Tuple[SlotName, SlotValue]]): List of visited slot and value pairs.

        Returns:
            Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]: 
            Updated error dictionary and visited pairs list.
        """

        error_dict = defaultdict(list)
        error_dict.setdefault('error', [])
        error_name, error_s_v_pairs = None, None

        for gold_slot, gold_value in delta_miss_gold.items():
            if (gold_slot, gold_value) in visited:
                continue        
            # Confused miss case: the predicted value is the same as the gold value, but the slot is different.
            if (gold_value in delta_over_pred.values()):            
                for (confused_slot, v) in delta_over_pred.items():
                    if v == gold_value and confused_slot != gold_slot:
                        error_name = 'delta_miss_confuse'
                        error_s_v_pairs = (gold_slot, gold_value, confused_slot, v)
                        error_dict, visited = self.record_error_and_update_visited(error_dict, error_name, error_s_v_pairs, visited)
            else:
                # if gold_slot in pred_delta_bs: 'delta_hall_val' error case. But we don't care about it here.
                # if gold_slot not in pred_delta_bs: 'delta_miss_total' error case.
                if gold_slot not in pred_delta_bs:
                    if error_dict.get('error') is None:
                        raise ValueError('Error case is None')
                    error_name = 'delta_miss_total'
                    error_s_v_pairs = (gold_slot, gold_value)
                
                # if gold_value is a special value and not in the prediction, record the error.
                if gold_value in self.special_values and pred_delta_bs.get(gold_slot, None) == None:
                    error_name = f'delta_miss_{re.sub(r"[^a-zA-Z]", "", gold_value)}'.lower()
                    error_s_v_pairs = (gold_slot, gold_value, gold_slot, prev_pred_bs.get(gold_slot, None))
                    
            error_dict, visited = self.record_error_and_update_visited(error_dict, error_name, error_s_v_pairs, visited) 
        return error_dict, visited

    def detect_delta_hallucinations(
        self,
        delta_miss_gold: MultiWOZDict,
        delta_over_pred: MultiWOZDict,
        gold_delta_bs: MultiWOZDict, 
        pred_delta_bs: MultiWOZDict,
        prev_gold_bs: MultiWOZDict,
        prev_pred_bs: MultiWOZDict,
        visited: List[Tuple[SlotName, SlotValue]]
    )-> Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]:
        """
        Detects hallucinated values in the prediction compared to the gold standard and previous predictions,
        and records errors.

        Args:
            delta_miss_gold (MultiWOZDict): Missing values in the gold standard.
            delta_over_pred (MultiWOZDict): Over-predicted values.
            gold_delta_bs (MultiWOZDict): The gold standard delta belief state.
            pred_delta_bs (MultiWOZDict): The predicted delta belief state.
            prev_pred_bs (MultiWOZDict): The previous predicted belief state.
            visited (List[Tuple[SlotName, SlotValue]]): List of visited slot and value pairs.

        Returns:
            Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]: 
            Updated error dictionary and visited pairs list.
            
        """
        error_dict = defaultdict(list)
        error_dict.setdefault('error', [])
        error_name, error_s_v_pairs, tmp_visited = None, None, []

        for pred_slot, pred_value in delta_over_pred.items():
            if (pred_slot, pred_value) in visited:
                continue
            
            if pred_slot in gold_delta_bs:
                error_name = 'delta_hall_val'
                error_s_v_pairs = (pred_slot, gold_delta_bs[pred_slot], pred_slot, pred_value)
                tmp_visited = visited

            elif pred_slot in prev_pred_bs:
                if pred_value == prev_pred_bs[pred_slot]:
                    continue    # parroting the previous slot, value pair 
                elif prev_gold_bs.get(pred_slot, None) and pred_value == prev_gold_bs[pred_slot]:
                    continue    # or correctly predicting the previous slot, value pair in the current turn
                if pred_value != prev_pred_bs[pred_slot]:
                    error_name = 'delta_hall_overwrite'
                    error_s_v_pairs = (pred_slot, pred_value)
            else:
                error_name = 'delta_hall_total'
                error_s_v_pairs = (pred_slot, pred_value)
            
            error_dict, visited = self.record_error_and_update_visited(error_dict, error_name, error_s_v_pairs, tmp_visited)

        for gold_slot, gold_value in delta_miss_gold.items():
            if (gold_slot, gold_value) in visited:
                continue
            
            if (gold_value not in delta_over_pred.values()):
                if gold_slot in pred_delta_bs:
                    error_name = f'delta_hall_val'
                    error_s_v_pairs = (gold_slot, gold_value, gold_slot, pred_delta_bs[gold_slot])
                
            error_dict, visited = self.record_error_and_update_visited(error_dict, error_name, error_s_v_pairs, visited) 
        
        return error_dict, visited

    def detect_error_propagations(
        self,
        delta_miss_gold, 
        delta_over_pred, 
        gold_bs, 
        pred_bs, 
        prev_gold_bs, 
        prev_pred_bs, 
        visited,
        prev_item: dict,
    )->Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]:
        """
        Analyzes the error cases that are propagated from the previous turn.

        Args:
            delta_miss_gold (MultiWOZDict): Missing values in the gold standard.
            delta_over_pred (MultiWOZDict): Over-predicted values.
            gold_bs (MultiWOZDict): The gold standard belief state.
            pred_bs (MultiWOZDict): The predicted belief state.
            prev_gold_bs (MultiWOZDict): The previous gold standard belief state.
            prev_pred_bs (MultiWOZDict): The previous predicted belief state.
            visited (List[Tuple[SlotName, SlotValue]]): List of visited slot and value pairs.
            prev_item (dict): The previous error log.
        
        Returns:
            Tuple[Dict[str, List[Tuple[str, Tuple[SlotName, SlotValue]]]], List[Tuple[SlotName, SlotValue]]]: 
            Updated error dictionary and visited pairs list

        """
        prev_over_pred = compute_dict_difference(prev_pred_bs, prev_gold_bs)
        prev_miss_gold = compute_dict_difference(prev_gold_bs, prev_pred_bs)
        
        full_over_pred = compute_dict_difference(pred_bs, gold_bs)
        full_miss_gold = compute_dict_difference(gold_bs, pred_bs)
        
        error_dict = defaultdict(list)
        for err_item in prev_item.get('error', []):
            if isinstance(err_item, tuple) and len(err_item) == 2:
                err_name, err_s_v = err_item
            else:
                print("Unexpected error format:", err_item)
        
            if 'hall' in err_name:
                error_slot, error_value = err_s_v[-2], err_s_v[-1]
            if 'miss' in err_name :
                error_slot, error_value = err_s_v[0], err_s_v[1]
                if 'delete' in err_name:
                    error_slot, error_value = err_s_v[-2], err_s_v[-1]
                
            if (error_slot, error_value) in visited:
                continue
            
            if (error_slot, error_value) in prev_miss_gold.items() or (error_slot, error_value) in prev_over_pred.items():
                if (error_slot, error_value) in full_over_pred.items() or (error_slot, error_value) in full_miss_gold.items():
                    if 'delete' in err_name:
                        prop_name = 'error_prop_'+'_'.join(err_name.split('_')[-2:])
                        error_dict, visited = self.record_error_and_update_visited(error_dict, prop_name, err_s_v, visited)
                    if (error_slot, error_value) in delta_miss_gold.items() or (error_slot, error_value) in delta_over_pred.items():
                        continue
                    prop_name = 'error_prop_'+'_'.join(err_name.split('_')[-2:])
                    error_dict, visited = self.record_error_and_update_visited(error_dict, prop_name, err_s_v, visited)
        return error_dict, visited
    
    def categorize_error_case(
        self, 
        item: dict, 
        prev_item: dict, 
        gold_bs: MultiWOZDict, 
        pred_bs: MultiWOZDict, 
        gold_delta_bs: MultiWOZDict, 
        pred_delta_bs: MultiWOZDict, 
        prev_gold_bs: MultiWOZDict, 
        prev_pred_bs: MultiWOZDict,
        **kwargs
    ) -> dict:
        """
        Categories the error cases into different types and records them in the log.
        
        Args:
            item (dict): The current error log.
            prev_item (dict): The previous error log.
            gold_bs (MultiWOZDict): The gold standard belief state.
            pred_bs (MultiWOZDict): The predicted belief state.
            gold_delta_bs (MultiWOZDict): The gold standard delta belief state.
            pred_delta_bs (MultiWOZDict): The predicted delta belief state.
            prev_gold_bs (MultiWOZDict): The previous gold standard belief state.
            prev_pred_bs (MultiWOZDict): The previous predicted belief state.
        
        Returns:
            dict: The log updated error cases.
        """
        
        delta_miss_gold = compute_dict_difference(gold_delta_bs, pred_delta_bs)
        delta_over_pred = compute_dict_difference(pred_delta_bs, gold_delta_bs)

        visited = [] if kwargs.get('visited') is None else kwargs.get('visited')

        # handle the case which is already found and recorded in the current turn
        for err_item in item.get('error', []):
            if isinstance(err_item, tuple) and len(err_item) == 2:
                err_name, err_s_v = err_item
            else:
                print("Unexpected error format:", err_item)
    
            if len(err_s_v) > 2:
                visited.append((err_s_v[-2], err_s_v[-1]))
            visited.append((err_s_v[0], err_s_v[1]))

        # handle the case which prediction missed in the current turn
        error_case, miss_visited = self.detect_delta_missings(
            delta_miss_gold=delta_miss_gold, 
            delta_over_pred=delta_over_pred, 
            gold_delta_bs=gold_delta_bs,
            pred_delta_bs=pred_delta_bs,
            prev_pred_bs=prev_pred_bs,
            visited=[]
        )
        item['error'].extend(error_case.get('error', []))

        # handle the case which is over-predicted in the current turn
        error_case, hall_visited = self.detect_delta_hallucinations(
            delta_miss_gold=delta_miss_gold,
            delta_over_pred=delta_over_pred,
            gold_delta_bs=gold_delta_bs,
            pred_delta_bs=pred_delta_bs, 
            prev_gold_bs=prev_gold_bs,
            prev_pred_bs=prev_pred_bs, 
            visited=[]
        )
        item['error'].extend(error_case.get('error', []))
        
        # To allow the same (slot, value) pair to be visited multiple times in the miss, hall error cases
        # But, not in the error propagation cases
        visited.extend(miss_visited)
        visited.extend(hall_visited)
        
        # handle the case which is propagated from the previous turn
        error_case, visited = self.detect_error_propagations(
            delta_miss_gold=delta_miss_gold, 
            delta_over_pred=delta_over_pred, 
            gold_bs=gold_bs, 
            pred_bs=pred_bs,
            prev_gold_bs=prev_gold_bs, 
            prev_pred_bs=prev_pred_bs,
            visited=visited, prev_item=prev_item
        )
        item['error'].extend(error_case.get('error', []))

        return item
    
    def reason_error(
        self,
        analyzed_item: dict,
    ):
        """
        대화 맥락을 기반으로 발생한 오류 사례의 원인을 분석합니다.

        Args:
            analyzed_item (dict): 오류를 포함한 대화의 단일 턴 로그.

        Returns:
            dict: 각 오류에 대한 원인이 추가된 업데이트된 로그.
        """

        reason_dict = {
            "delta_hall_total": "This error is occred because the model hallucinated the slot and value. ",
            "delta_hall_overwrite": "This error is occred because the model updated the slot and value with incorrect value. ",
            "delta_hall_val": "This error is occred because the model correctly captures slot but hallucinated the value. ",

            "delta_miss_total": "This error is occred because the model missed the slot and value in the dialog. ",            
            "delta_miss_confuse": "This error is occred because the model confused the slot with another slot. ",
            "delta_miss_delete": "This error is occred because the model missed the slot and value which was deleted. ",
            "delta_miss_dontcare": "This error is occred because the model missed the user's flexible preference. ",
        }

        def retrun_explanation_dict(error_type, error_slot_value):
            
            if 'delta_hall' in error_type:
                slot_name, value_name = error_slot_value[-2], error_slot_value[-1]
                dicts = {
                    'annotation': f"This means that the model made '{error_type }' error because the provided annotation was incorrect considering dialog.",
                    'context_hallucination': f"This means that the model made '{error_type }' error because it captures a {slot_name} and {value_name} that are irrelevant to the current dialogue, even though they are mentioned in the dialogue context.",
                    'hallucination': f"This means that the model made '{error_type }' error because it hallucinated {slot_name} and {value_name} which are completely outside the context of the dialogue.",
                    'infer_hallucination': f"This means that the model made '{error_type }' error because it inferred the {slot_name} and {value_name} incorrectly from the dialogue context.",
                    'intent_hallucination': f"This means that the model made '{error_type }' error because it misunderstood the user's intent from the utterances.",
                    'late': f"This means that the model made '{error_type }' error because it should have been predicted in earlier turns.",
                    'update_hallucination': f"This means that the model made '{error_type }' error because it updated the {slot_name} and {value_name} with incorrect slot or value.",
                    'co_reference': f"This means the model made '{error_type }' error because it failed to resolve the co-reference using the dialogue context and current utterances.",
                    'time': f"This means that the model made '{error_type }' error because it captures the wrong time.",
                }
                
            if 'delta_miss' in error_type:
                slot_name, value_name = error_slot_value[-2], error_slot_value[-1]
                dicts =  {
                    'annotation': f"This means that the model made '{error_type }' error because the provided annotation was incorrect considering dialog.",
                    'miss_utter': f"This means that the model made '{error_type }' error because it misses user's flexible preference from the utterances.",
                    'context_change': f"This means that the model made '{error_type }' error because it misses the context change in dialog context.",
                    'co-reference': f"This means that the model made '{error_type }' error because it failed to resolve the co-reference using the dialogue context and current utterances.",
                    'usr_ex_confirm': f"This means that the model made '{error_type }' error because it misses the {slot_name} and {value_name} which the user explicitly confirmed.",
                    'usr_im_confirm': f"This means that the model made '{error_type }' error because it misses the {slot_name} and {value_name} which the user implicitly confirmed.",
                    'usr_request': f"This means that the model made '{error_type }' error because it misses the {slot_name} and {value_name} which the user requested.",
                    'usr_state': f"This means that the model made '{error_type }' error because it misses the {slot_name} and {value_name} which the user's state.",
                    'usr_refusal': f"This means that the model made '{error_type }' error because it misses the user's refusal from the utterances.",
                    'domain_slot_miss_align': f"This means that the model made '{error_type }' error because the {slot_name} and {value_name} are not aligned correctly to the domain.",
                    'slot_value_miss_align': f"This means that the model made '{error_type }' error because the value is not aligned correctly to the slot.",
                    'hallucination': f"This means that the model made '{error_type }' error because it hallucinated {slot_name} and {value_name} which are completely outside the context of the dialogue.",
                    'mixed_intent': f"This means that the model made '{error_type }' error because it misses mixed the intents from the user's utterances.",
                    'slot_confuse': f"This means that the model made '{error_type }' error because it confused the slot with another slot.",            
                    'context_hallucination': f"This means that the model made '{error_type }' error because it captures a {slot_name} and {value_name} that are irrelevant to the current dialogue, even though they are mentioned in the dialogue context.",
                }            
            return dicts
    
        # 분석된 항목의 정보를 가져옵니다.
        context = analyzed_item.get('dialog')
        gold_bs = analyzed_item.get('turn_slot_values')
        pred_bs = analyzed_item.get(f'pred_{self.parsing_func.__name__}')
        analyzed_item['error_reason'] = copy.deepcopy(analyzed_item.get('error', []))

        # iterate over the error cases
        for i, (error_type, error_slot_value) in enumerate(analyzed_item.get('error', [])):
            
            if 'error_prop' in error_type:
                analyzed_item['error_reason'][i] = (error_type, error_slot_value, 'error_propagation')
                continue
                
            prompt = [{"role": "system", "content": "You are a talented error analyzer! Let's analyze the error cases together."}]
            
            user_prompt = f"### Dialog ###\n"
            for sys_utt, usr_utt in zip(context['sys'], context['usr']):
                user_prompt += f"**System**: {sys_utt}\n"
                user_prompt += f"**User**: {usr_utt}\n"

            user_prompt += f"### Gold Standard Dialogue State Change ###\n"
            user_prompt += f"    {gold_bs}\n"
            user_prompt += f"### Predicted Dialogue State Change ###\n"
            user_prompt += f"    {pred_bs}\n"

            user_prompt += f"### Error Type: {error_type} ###\n"
            user_prompt += "    " + reason_dict[error_type]

            explanation_list = list(retrun_explanation_dict(error_type, error_slot_value).items())

            user_prompt += "The reason for the error is as follows: \n"
            for idx,(reason_name, explanaton) in enumerate(explanation_list):
                user_prompt += f"({idx}) **{reason_name}**\n"
                user_prompt += f"{explanaton} \n\n"

            user_prompt += "#### Instruction ####\n"
            user_prompt += f"   - Based on the dialogue, gold standard dialogue state change, and predicted dialogue state change, select the most related and correct reason for the error. \n"
            user_prompt += f"   - DO NOT select reasons unrelated to the error. \n"
            user_prompt += f"   - DO NOT generate text. \n"
            user_prompt += f"   - DO NOT generate multiple scalar values.\n\n"
            user_prompt += f"## Provide only a single scalar value as output. ##\n"

            prompt.append({"role": "user","content": user_prompt})

            # Make a request to the LLM to generate the reason for the error.
            prompt_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
            prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)[0]
            try:
                result = self.LLM.generate(prompts, sampling_params=self.sampling_params)

                completions = result[0].outputs[0].text
                scalars = re.sub(r"[^0-9 ]", "", completions)
                scalars = int(scalars)

                reason = [explanation_list[s][0] for s in scalars]
            except Exception as e:
                reason = None
                print(f"Error in generating the reason for the error: {e}")

            analyzed_item['error_reason'][i] = (error_type, error_slot_value, reason)

        return analyzed_item
    
    def analyze(self):
        """
        Analyzes the errors in the prediction compared to the gold standard and records them.
        """
        print('Start analyzing the error cases...')
        logs = read_json(self.result_file_path)
        
        analyzed_log = []
        n_correct = 0
        prev_item = {}
        for idx, data_item in tqdm(enumerate(logs), desc="analyzing items", total=len(logs)):
            if data_item['turn_id'] == 0:
                prev_item = {}            

            analyzed_item = copy.deepcopy(data_item)
            
            (   
                gold_bs, pred_bs, 
                gold_delta_bs, pred_delta_bs, 
                prev_gold_bs, prev_pred_bs, error_reason
            ) = self.preprocess_belief_state(analyzed_item, prev_item)
            
            if pred_bs==gold_bs:
                n_correct+=1

            analyzed_item['error'] = []

            if error_reason:
                analyzed_item['error'].append((error_reason, None)) 
                continue

            analyzed_item = self.categorize_error_case(
                item=analyzed_item, prev_item=prev_item, 
                gold_bs=gold_bs, 
                pred_bs=pred_bs, 
                gold_delta_bs=gold_delta_bs, 
                pred_delta_bs=pred_delta_bs, 
                prev_gold_bs=prev_gold_bs, 
                prev_pred_bs=prev_pred_bs
            )
            
            # remove the redundant error cases
            analyzed_item['error'] = sorted(list(set(tuple(x) for x in analyzed_item['error'])))

            analyzed_item = self.reason_error(analyzed_item)
            
            analyzed_item = sort_data_item(data_item=analyzed_item, parsing_func=self.parsing_func.__name__)
            analyzed_log.append(analyzed_item)
            prev_item = analyzed_item

            if idx % 1000 == 0:
                save_analyzed_log(output_dir_path=self.output_dir_path, analyzed_log=analyzed_log)

        save_analyzed_log(output_dir_path=self.output_dir_path, analyzed_log=analyzed_log)
        self.plot_error_stats(analyzed_log)

        return analyzed_log
    
    def plot_error_stats(self, analyzed_log):
        # Count the error statistics
        error_stats = defaultdict(int)
        for data_item in analyzed_log:
            for error_name, error_s_v_pairs in data_item['error']:
                error_stats[error_name] += 1

        # Plot the error statistics wiht exact numbers
        error_stats = dict(sorted(error_stats.items(), key=lambda x: x[1], reverse=True))
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(20, 10))
        sns.barplot(x=list(error_stats.values()), y=list(error_stats.keys()))
        plt.xlabel('Number of Errors')
        plt.ylabel('Error Type')
        plt.grid(axis='y')
        plt.title('Error Statistics')

        for index, (key, value) in enumerate(error_stats.items()):
            plt.text(value, index,
                    str(value))
        
        # save the plot
        plt.savefig(self.output_dir_path + '/error_stats.png')

        error_reason_stats = defaultdict(int)
        for data_item in analyzed_log:
            for error_name, error_s_v_pairs, reason in data_item['error_reason']:
                if reason is None:
                    continue
                for r in reason:
                    error_reason_stats[r] += 1

        # Plot the error statistics wiht exact numbers
        error_reason_stats = dict(sorted(error_reason_stats.items(), key=lambda x: x[1], reverse=True))
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(20, 10))
        sns.barplot(x=list(error_reason_stats.values()), y=list(error_reason_stats.keys()))
        plt.xlabel('Number of Errors')
        plt.ylabel('Error Type')
        plt.grid(axis='y')
        plt.title('Error Statistics')

        for index, (key, value) in enumerate(error_reason_stats.items()):
            plt.text(value, index,
                    str(value))
        
        # save the plot
        plt.savefig(self.output_dir_path + '/error_reason_stats.png')

        return error_stats

    def show_stats(self, analyzed_log):
        total_acc, total_f1 = 0, 0
        jga_by_turn_id = defaultdict(list)  # use to record the accuracy
        jga_by_dialog = defaultdict(list)  # use to record the accuracy
        wrong_smaples = []
        n_correct = 0
        n_total = len(analyzed_log)
        for data_item in analyzed_log:
            pred = data_item['pred']
            this_jga, this_acc, this_f1 = evaluate(pred, data_item['slot_values'])
            total_acc += this_acc
            total_f1 += this_f1
            if this_jga:
                n_correct += 1
                jga_by_turn_id[data_item['turn_id']].append(1)
                jga_by_dialog[data_item['ID']].append(1)
            else:
                jga_by_turn_id[data_item['turn_id']].append(0)
                jga_by_dialog[data_item['ID']].append(0)
                wrong_smaples.append(data_item)

        stats = evaluate_logs(analyzed_log, test_set=analyzed_log)
        slot_prf = slot_level_f1(analyzed_log, tp_means_correct=True)

        slot_acc: Dict[str, Counter] = defaultdict(Counter)
        for turn in tqdm(analyzed_log, desc="calculating slot-level F1", total=len(analyzed_log)):
            for gold_slot, gold_value in turn['slot_values'].items():
                slot_acc[gold_slot]['total'] += 1
                if gold_slot in turn['pred'] and turn['pred'][gold_slot] == gold_value:
                    slot_acc[gold_slot]['right'] += 1

        slot_acc = {slot: slot_acc[slot]['right'] / slot_acc[slot]['total'] for slot in slot_acc}
        slot_acc = dict(sorted(slot_acc.items(), key=lambda x: x[0]))
        
        slot_f1 = {k: v[1] for k, v in slot_prf.items()}
        slot_f1 = dict(sorted(slot_f1.items(), key=lambda x: x[0]))

        stats_df = pd.DataFrame(slot_acc.items(), columns=['slot', 'acc'])
        stats_df.merge(pd.DataFrame(slot_f1.items(), columns=['slot', 'f1']), on='slot')

        print(stats_df)
        stats_df.to_csv(self.output_dir_path + '/slot_acc_f1.csv', index=False)
        
        return None

    def make_confusion_matrix(self, analyzed_log, criteria='value'):
        """
        Predicted State Change
        """
        slots = list(self.normalizer.ontology.valid_slots)
        pred_delta_confusion_matrix = {
            gold_slot: {pred_slot:0 for pred_slot in slots} 
            for gold_slot in slots+['hall_value', 'hall_total']
        }
        pred_delta_confusion_matrix['text_hallucination'] = 0

        gold_delta_confusion_matrix = {
            gold_slot: {pred_slot:0 for pred_slot in slots+['miss_value', 'miss_total']} 
            for gold_slot in slots
        }

        for data_item in analyzed_log:
            
            pred_delta_bs = data_item[f'pred_delta_{self.parsing_func.__name__}']
            gold_delta_bs = data_item['turn_slot_values']

            # filter out text hallucination totally wrong generation
            if pred_delta_bs == {} and 'update' not in data_item['completion']: 
                for slot_name in gold_delta_bs:
                    pred_delta_confusion_matrix['text_hallucination'] += 1
                    continue
            
            if criteria == 'value':
                pred_delta_confusion_matrix, gold_delta_confusion_matrix = self.update_conf_mat_value(
                    pred_delta_bs, gold_delta_bs, 
                    pred_delta_confusion_matrix, gold_delta_confusion_matrix
                )
            elif criteria == 'slot':
                pred_delta_confusion_matrix, gold_delta_confusion_matrix = self.update_conf_mat_value(
                    pred_delta_bs, gold_delta_bs, 
                    pred_delta_confusion_matrix, gold_delta_confusion_matrix
                )
            else:
                raise ValueError('Invalid criteria')
                    
        return pred_delta_confusion_matrix, gold_delta_confusion_matrix

    def update_conf_mat_value(
        self, 
        pred_delta_bs, gold_delta_bs,
        pred_delta_confusion_matrix, 
        gold_delta_confusion_matrix
    ):
        for pred_slot, pred_value in pred_delta_bs.items():
            if pred_value in list(gold_delta_bs.values()):
                for gold_slot in [k for k, v in gold_delta_bs.items() if v == pred_value]:
                    pred_delta_confusion_matrix[gold_slot][pred_slot] += 1
            else:
                if pred_slot in list(gold_delta_bs.keys()):
                    # value hall: pred: {gold_slot: wrong_value}
                    pred_delta_confusion_matrix['hall_value'][pred_slot] += 1
                else:
                    # hall_total: pred: {wrong_slot: wrong_value}
                    pred_delta_confusion_matrix['hall_total'][pred_slot] += 1

        for gold_slot, gold_value in gold_delta_bs.items():
            if gold_value in list(pred_delta_bs.values()):
                for pred_slot in [k for k, v in pred_delta_bs.items() if v == gold_value]:
                    gold_delta_confusion_matrix[gold_slot][pred_slot] += 1
            else:
                if gold_slot in list(pred_delta_bs.keys()):
                    # value miss: pred:{gold_slot: wrong_value}
                    gold_delta_confusion_matrix[gold_slot]['miss_value'] += 1
                else:
                    # miss_total: pred: {wrong_slot: wrong_value}
                    gold_delta_confusion_matrix[gold_slot]['miss_total'] += 1
        
        return pred_delta_confusion_matrix, gold_delta_confusion_matrix
    
    def _update_conf_mat_slot(
        self,
        pred_delta_bs, gold_delta_bs,
        pred_delta_confusion_matrix,
        gold_delta_confusion_matrix
    ):
        for pred_slot, pred_value in pred_delta_bs.items():
            if pred_slot not in list(gold_delta_bs.keys()):
                if pred_value in list(gold_delta_bs.values()):
                    for gold_slot in [k for k, v in gold_delta_bs.items() if v == pred_value and k != pred_slot]:
                        pred_delta_confusion_matrix[gold_slot][pred_slot] += 1
                else:
                    pred_delta_confusion_matrix['hall_total'][pred_slot] += 1
            else:
                if pred_value == gold_delta_bs[pred_slot]:
                    pred_delta_confusion_matrix[pred_slot][pred_slot] += 1
                else:
                    pred_delta_confusion_matrix['hall_value'][pred_slot] += 1
                
        for gold_slot, gold_value in gold_delta_bs.items():
            if gold_slot in list(pred_delta_bs.keys()):
                if gold_value == pred_delta_bs[gold_slot]:
                    gold_delta_confusion_matrix[gold_slot][gold_slot] += 1
                else:
                    gold_delta_confusion_matrix[gold_slot]['miss_value'] += 1
            else:
                if gold_value in list(pred_delta_bs.values()):
                    for pred_slot in [k for k, v in pred_delta_bs.items() if v == gold_value and k != gold_slot]:
                        gold_delta_confusion_matrix[gold_slot][pred_slot] += 1
                else:
                    gold_delta_confusion_matrix[gold_slot]['miss_total'] += 1

        return pred_delta_confusion_matrix, gold_delta_confusion_matrix

    def show_state_change_confusion_matrix(self):
        analyzed_log = load_analyzed_log(self.output_dir_path)
        pred_delta_conf_mat, gold_delta_conf_mat = self.make_confusion_matrix(analyzed_log, criteria='value')

        slots = list(self.normalizer.ontology.valid_slots)
        # Convert the confusion matrix to the pandas DataFrame
        fig, axes = plt.subplots(1,2, figsize=(30, 15))
        pred_delta_df = pd.DataFrame(pred_delta_conf_mat).T.drop('text_hallucination', axis=0).dropna()
        gold_delta_df = pd.DataFrame(gold_delta_conf_mat).T.dropna()

        sns.heatmap(pred_delta_df, mask=(pred_delta_df == 0), annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black', ax=axes[0])
        sns.heatmap(gold_delta_df, mask=(gold_delta_df == 0), annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black', ax=axes[1])

        axes[0].set_xlabel('Predicted State Change')
        axes[0].set_ylabel('Gold State Change')
        axes[1].set_xlabel('Predicted State Change')
        axes[1].set_ylabel('Gold State Change')

        print('Confusion matrix saved to the output directory')
        print('PRED Confusion matrix')
        print("# of hal value (Right SLOT Wrong VALUE): ", pred_delta_df.loc['hall_value'].sum())
        print("# of confused (Wrong SLOT Right VALUE): ", pred_delta_df.loc[slots].values.sum() - pred_delta_df.loc[slots].values.diagonal().sum())
        print("# of hall total (Wrong SLOT Wrong VALUE): ", pred_delta_df.loc['hall_total'].sum())
        print()
        print('GOLD Confusion matrix')
        print("# of miss value (Right SLOT Wrong VALUE): ", gold_delta_df.loc[:, 'miss_value'].sum())
        print("# of confused (Wrong SLOT Right VALUE): ", gold_delta_df.loc[slots].values.sum() - gold_delta_df.loc[slots].values.diagonal().sum())
        print("# of miss total (Wrong SLOT Wrong VALUE): ", gold_delta_df.loc[:, 'miss_total'].sum())
        
        plt.savefig(self.output_dir_path + '/confusion_matrix.png')
        return None
    

if __name__ == '__main__':

    llm_config = {
        "engine":"meta-llama/Meta-Llama-3-8B-Instruct",
        "quantization": None,
    }
    analyzer = ErrorAnalyzer(
        train_data_path='/home/haesungpyun/my_refpydst/data/mw21_1p_train_v1.json',
        result_file_path='/home/haesungpyun/my_refpydst/outputs/runs/table4/5p/smapling_exp/split_v1_topk_bm_5_fs_5/running_log.json',
        output_dir_path='/home/haesungpyun/my_refpydst/outputs/runs/table4/5p/smapling_exp/split_v1_topk_bm_5_fs_5',
        use_llm=False,
        llm_config=llm_config,
        parsing_func='naive_iterative_parsing'
    )
    analyzer.analyze()

    # # analyzer.show_state_change_confusion_matrix()
    # output_dir = 'outputs/runs/table4/5p/smapling_exp/split_v1_topk_bm_5_fs_5/'
    # analyzed_log = load_analyzed_log(output_dir_path=output_dir)

    # # Count the error statistics
    # error_stats = defaultdict(int)
    # for data_item in analyzed_log:
    #     for error_name, error_s_v_pairs in data_item['error']:
    #         error_stats[error_name] += 1

    # # Plot the error statistics
    # error_stats = dict(sorted(error_stats.items(), key=lambda x: x[1], reverse=True))
    # sns.set_theme(style="whitegrid")
    # plt.figure(figsize=(20, 10))
    # sns.barplot(x=list(error_stats.values()), y=list(error_stats.keys()))
    # plt.grid(axis='y')
    # plt.xlabel('Number of Errors')
    # plt.ylabel('Error Type')
    # plt.title('Error Statistics')

    # for index, (key, value) in enumerate(error_stats.items()):
    #     plt.text(value, index,
    #             str(value))
    
    # # save the plot
    # plt.savefig(output_dir + '/error_stats.png')
