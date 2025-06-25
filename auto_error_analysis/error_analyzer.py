import abc
import os
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


from normalization.data_ontology_normalizer import DataOntologyNormalizer
from db.ontology import Ontology
from data_types import SlotName, SlotValue, MultiWOZDict
from utils import validate_path_and_make_abs_path,read_json, save_analyzed_log, load_analyzed_log
from bs_utils import compute_dict_difference, sort_data_item, unroll_or, update_dialogue_state
from plot_generator import PlotGenerator
from reason_generator import ReasonGenerator

from completion_parser import PARSING_FUNCTIONS

import torch

domain_slot_dict = {
    'attraction': ['area', 'name', 'type'], 
    'hotel': ['area', 'book day', 'book people', 'book stay', 'internet', 'name', 'parking', 'pricerange', 'stars', 'type'], 
    'restaurant': ['area', 'book day', 'book people', 'book time', 'food', 'name', 'pricerange'], 
    'taxi': ['arriveby', 'departure', 'destination', 'leaveat'], 
    'train': ['arriveby', 'book people', 'day', 'departure', 'destination', 'leaveat']
}
slot_list = [
    'area', 'name', 'type', 'book day', 'book people', 'book stay', 'internet', 'parking', 'pricerange', 
    'stars', 'food', 'book time', 'arriveby', 'day', 'departure', 'destination', 'leaveat']

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
                                     [error_analysis_parse_nl_completion(default), error_analysis_iterative_parsing, 
                                     iterative_parsing, parse_python_modified, parse_state_change, parse_python_completion]
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
        
        self.plot_generator = PlotGenerator(output_dir_path=self.output_dir_path)

        self.special_values = special_values or ['dontcare', '[DELETE]']

        self.use_llm = use_llm
        
        if self.use_llm:     
            if not llm_config:
                llm_config = {
                    "engine":"../models/Meta-Llama-3-70B-Instruct-GPTQ",
                    "quantization":"GPTQ",
                }
            self.reason_generator = ReasonGenerator(llm_config=llm_config)

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
        prev_pred_bs = prev_item.get(f'pred_{self.parsing_func.__name__}', {})
        analyzed_item[f'last_pred_{self.parsing_func.__name__}'] = prev_pred_bs
        tmp_prev_pred_bs = copy.deepcopy(prev_pred_bs) if prev_pred_bs else {}
        prev_gold_bs, prev_pred_bs = unroll_or(gold=analyzed_item['last_slot_values'], pred=tmp_prev_pred_bs)
    
        # Delta(State Change) Belief State parsing. Parse the completion and normalize considering surface forms
        return_tuple = self.parsing_func(analyzed_item['completion'], state=tmp_prev_pred_bs)
        if isinstance(return_tuple, tuple) and len(return_tuple) == 2:
            parsed_pred_delta_bs, error_reason = return_tuple
        else:
            parsed_pred_delta_bs, error_reason = return_tuple, None
        if parsed_pred_delta_bs:
            parsed_pred_delta_bs = self.normalizer.normalize(raw_parse=parsed_pred_delta_bs) if 'DELETE' not in str(parsed_pred_delta_bs) else parsed_pred_delta_bs
        analyzed_item[f'pred_delta_{self.parsing_func.__name__}'] = parsed_pred_delta_bs
        
        tmp_pred_delta = copy.deepcopy(parsed_pred_delta_bs) if parsed_pred_delta_bs else {}
        gold_delta_bs, tmp_pred_delta = unroll_or(gold=analyzed_item['turn_slot_values'], pred=tmp_pred_delta)
        
        # Accumulated Dialogue State Belief State. Update the pred with parsed delta(State Change) belief state
        pred_bs = update_dialogue_state(context=tmp_prev_pred_bs, normalized_turn_parse=tmp_pred_delta) 
        gold_bs, pred_bs = unroll_or(gold=analyzed_item['slot_values'], pred=pred_bs)
        if not prev_pred_bs and not parsed_pred_delta_bs:
            pred_bs = None
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
            
            if err_name == 'format error':
                continue
        
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
    

    def analyze_error_details(self, analyzed_log: dict):
        for idx, data_item in tqdm(enumerate(analyzed_log), desc="analyzing items", total=len(analyzed_log)):

            context = data_item.get('dialog')
            gold_delta_bs = data_item.get('turn_slot_values')
            pred_delta_bs = data_item.get(f'pred_delta_{self.parsing_func.__name__}')
            prev_pred_bs = data_item.get(f'last_pred_{self.parsing_func.__name__}')

            if data_item.get('error') == [] or data_item.get('error')[0][1] is None:
                continue
            
            data_item['error_reason'] = [None] * len(data_item.get('error', []))
            for i, (error_name, error_s_v_pairs) in enumerate(data_item.get('error', [None, None])):
                if 'error_prop' in error_name:
                    data_item['error_reason'][i] = (error_name, error_s_v_pairs, 'error_propagation','error_propagation','error_propagation')
                    continue
                elif 'format error' in error_name:
                    data_item['error_reason'][i] = (error_name, error_s_v_pairs, 'format_error','format_error','format_error')
                    continue
                
                error_detail = None

                if error_name == 'delta_miss_confuse' or error_name == 'delta_hall_val':

                    gold_slot, gold_value, confused_slot, confused_value = error_s_v_pairs
                    if confused_slot in gold_delta_bs:
                        error_detail = "confuse slot"
                    else:
                        if confused_slot in pred_delta_bs:
                            if pred_delta_bs[confused_slot] == gold_value:
                                error_detail = "repeat slot value in context"
                            else:
                                error_detail = "update correct slot-value in context to wrong slot-value"
                        else:
                            if confused_value in prev_pred_bs.values():
                                error_detail = "incorrect co-reference resolution"
                            else:
                                error_detail = "totally hallucinate slot-value"

                elif error_name == 'delta_miss_total':
                    error_detail = "totally miss slot-value"

                elif error_name == 'delta_miss_delete':
                    gold_slot, gold_value, pred_slot, pred_value = error_s_v_pairs
                    if pred_value is None:
                        error_detail = "error_propagation"
                    else:
                        error_detail = "predict correct slot but hallucinate value"

                elif error_name == 'delta_miss_dontcare':
                    error_detail = "miss dontcare"

                elif error_name == 'delta_hall_overwrite':
                    error_detail = "update correct slot-value in context to wrong slot-value"

                elif error_name == 'delta_hall_total':
                    error_detail = ""
                    pred_slot, pred_value = error_s_v_pairs
                    domain, slot = pred_slot.split('-')
                    if prev_pred_bs is not None and pred_value in prev_pred_bs.values():
                        error_detail = "incorrect co-reference resolution"
                    else:
                        error_detail = "extract wrong value from current utterance"
                    
                    utt_str = f"sys: {data_item['dialog']['sys'][-1]} usr: {data_item['dialog']['usr'][-1]}" 
                    
                    in_schema, in_utt = 0, 0
                    if domain in utt_str:
                        in_utt = 1
                    else:
                        if self.use_llm:
                            in_utt = self.reason_generator.word_in_sentence(utt_str, domain)
                        else:
                            in_utt = 0
                    
                    if domain in domain_slot_dict:
                        in_schema = 1
                    else:
                        in_schema = 0
                    
                    if in_utt == 1 and in_schema == 1:
                        error_detail += ", capture correct domain from dialogue"
                    elif in_utt == 1 and in_schema == 0:
                        error_detail += ", cannot remember domain from schema"
                    elif in_utt == 0 and in_schema == 1:
                        error_detail += ", misunderstand domain from dialogue"
                    else:
                        error_detail += ", totally hallucinate domain"
                    
                    in_utt, in_schema = 0, 0
                    if slot in utt_str:
                        in_utt = 1
                    else:
                        if self.use_llm:
                            in_utt = self.reason_generator.word_in_sentence(utt_str, slot)
                        else:
                            in_utt = 0
                    
                    if domain in domain_slot_dict:
                        if slot in domain_slot_dict[domain]:
                            in_schema = 1
                        else:
                            in_schema = 0
                    else:
                        if slot in slot_list:
                            in_schema = 1
                        else:
                            in_schema = 0
                    
                    if in_utt == 1 and in_schema == 1:
                        error_detail += ", capture correct slot from dialogue"
                    elif in_utt == 1 and in_schema == 0:
                        error_detail += ", cannot remember slot from schema"
                    elif in_utt == 0 and in_schema == 1:
                        error_detail += ", misunderstand slot from dialogue"
                    else:
                        error_detail += ", totally hallucinate slot"

                    # error_detail = tuple(error_detail.split(','))

                if self.use_llm:
                    action, target, analysis = self.reason_generator.identify_action_target(
                        context=context, gold_delta_bs=gold_delta_bs, 
                        pred_delta_bs=pred_delta_bs, prev_pred_bs=prev_pred_bs,
                        error_name=error_detail, error_s_v_pairs=error_s_v_pairs,
                    )
                data_item['error_reason'][i] = (error_name, error_s_v_pairs, error_detail, (action, target), analysis)

        return analyzed_log
    
    def analyze(self):
        """
        Analyzes the errors in the prediction compared to the gold standard and records them.
        """
        print('\n\n=============== Start analyzing the error cases... ===============\n\n')
        logs = read_json(self.result_file_path)
        
        analyzed_log = []
        n_correct = 0
        prev_item = {}
        for idx, data_item in tqdm(enumerate(logs), desc="analyzing items", total=len(logs)):
            if data_item['turn_id'] == 0:
                prev_item = {}            
            
            analyzed_item = copy.deepcopy(data_item)
            if isinstance(data_item['pred'], list):
                analyzed_item['pred'] = analyzed_item['pred'][0]
            if isinstance(data_item['pred_delta_slot_values'], list):
                analyzed_item['pred_delta_slot_values'] = analyzed_item['pred_delta_slot_values'][0]
            if isinstance(data_item['pred_prior_context'], list):
                analyzed_item['pred_prior_context'] = analyzed_item['pred_prior_context'][0]
            if isinstance(data_item['completion'], list):
                analyzed_item['completion'] = analyzed_item['completion'][0]

            (   
                gold_bs, pred_bs, 
                gold_delta_bs, pred_delta_bs, 
                prev_gold_bs, prev_pred_bs, error_reason
            ) = self.preprocess_belief_state(analyzed_item, prev_item)
            
            if pred_bs==gold_bs:
                n_correct+=1

            analyzed_item['error'] = []
            analyzed_item['error_reason'] = []

            if pred_bs is None:
                pred_bs = {}
            if pred_delta_bs is None:
                pred_delta_bs = {}
            if prev_pred_bs is None:
                prev_pred_bs = {}         

            if error_reason:
                analyzed_item['error'].append((error_reason, None)) 
                analyzed_item['error_reason'].append((error_reason, None, error_reason, error_reason, error_reason)) 
                error_case, visited = self.detect_error_propagations(
                    delta_miss_gold=gold_delta_bs, delta_over_pred={}, 
                    gold_bs=gold_bs, pred_bs=pred_bs,
                    prev_gold_bs=prev_gold_bs, prev_pred_bs=prev_pred_bs,
                    visited=[], prev_item=prev_item
                )
                analyzed_item['error'].extend(error_case.get('error', []))
                analyzed_item = sort_data_item(data_item=analyzed_item, parsing_func=self.parsing_func.__name__)
                analyzed_log.append(analyzed_item)
                prev_item = analyzed_item
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
            
            analyzed_item = sort_data_item(data_item=analyzed_item, parsing_func=self.parsing_func.__name__)
            analyzed_log.append(analyzed_item)
            prev_item = analyzed_item

            if idx % 1000 == 0:
                save_analyzed_log(output_dir_path=self.output_dir_path, analyzed_log=analyzed_log)

        analyzed_log = self.analyze_error_details(analyzed_log=analyzed_log)
        save_analyzed_log(output_dir_path=self.output_dir_path, analyzed_log=analyzed_log)

        self.plot_generator.plot_all_stats(analyzed_log)
        
        print('\n\n=============== Finished analyzing the error cases! ===============\n\n')
        print(f"Total number of turns: {len(analyzed_log)}")
        print(f"Number of correct turns: {n_correct}")
        print(f"Joint Goal Accuracy: {n_correct/len(analyzed_log)}\n\n")
        return analyzed_log
    

if __name__ == '__main__':
    import sys  
    try:
        output_dir_path = sys.argv[1]
    except:
        output_dir_path = '/home/haesungpyun/my_refpydst/final/error_analysis/8B/plain_text/random'
    try:
        parsing_func = sys.argv[2]
    except:
        # parsing_func = 'error_analysis_iterative_parsing'
        parsing_func = 'error_analysis_parse_nl_completion'

    llm_config = {
        "engine":"../models/Meta-Llama-3-70B-Instruct-GPTQ",
        "quantization":"GPTQ",
    }
    
    analyzer = ErrorAnalyzer(
        train_data_path='/home/haesungpyun/my_refpydst/data/mw21_0p_train.json',
        result_file_path=output_dir_path+'/running_log.json',
        output_dir_path=output_dir_path,
        use_llm=True,
        llm_config=llm_config,
        parsing_func=parsing_func
    )
    analyzed_log = analyzer.analyze()