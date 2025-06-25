import copy

from data_types import MultiWOZDict

def compute_dict_difference(A: MultiWOZDict, B: MultiWOZDict) -> MultiWOZDict:
    """
    Computes the difference between two dictionaries A and B, returning a dictionary
    containing key-value pairs present in A but not in B.

    Args:
        A (MultiWOZDict): First dictionary to compare.
        B (MultiWOZDict): Second dictionary to compare.

    Returns:
        MultiWOZDict: Dictionary containing differences from A compared to B.

    Raises:
        ValueError: If either A or B is not a dictionary.
    """
    if not isinstance(A, dict) or not isinstance(B, dict):
        raise ValueError('A and B must be dictionaries')
    
    return {key: value for key, value in A.items() if key not in B or B[key] != value}

# Sort the dictionary by keys or values
def _sort_dictionary(dictionary: dict, sort_by_key: bool = True) -> dict:
    """
    Sort a dictionary by its keys or values.

    Parameters:
    dictionary (dict): The dictionary to sort.
    sort_by_key (bool): If True, sort by keys; if False, sort by values in descending order.

    Returns:
    dict: The sorted dictionary.
    """
    if sort_by_key:
        return dict(sorted(dictionary.items(), key=lambda item: item[0]))
    else:
        return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

def sort_data_item(data_item, parsing_func='iterative_parsing'):
    """
    The data_item is a dictionary with keys as follows:
        ['ID', 'turn_id', 'domains', 'dialog', 'slot_values', 'turn_slot_values', 'last_slot_values', 
        'prompt', 'pred', 'pred_delta_slot_values', 'pred_prior_context', 'completion', 'all_completions', 
        'num_solutions', 'prompt_counts', 'examples', 'iter_parse_pred_delta', 'rights', 'error']
    """
    return_data_item = {}

    return_data_item['ID'] = data_item['ID']
    return_data_item['turn_id'] = data_item['turn_id']
    return_data_item['domains'] = data_item['domains']
    return_data_item['dialog'] = data_item['dialog']
    return_data_item['slot_values'] = _sort_dictionary(data_item['slot_values'])
    return_data_item[f'pred_{parsing_func}'] = _sort_dictionary(data_item[f'pred_{parsing_func}']) if data_item[f'pred_{parsing_func}'] else None

    return_data_item['turn_slot_values'] = _sort_dictionary(data_item['turn_slot_values'])
    return_data_item[f'pred_delta_{parsing_func}'] = _sort_dictionary(data_item[f'pred_delta_{parsing_func}']) if data_item[f'pred_delta_{parsing_func}'] else None
    return_data_item['completion'] = data_item['completion']

    return_data_item['error'] = data_item['error']
    return_data_item['error_reason'] = data_item.get('error_reason', None)
 
    # return_data_item['all_completions'] = data_item['all_completions']

    return_data_item['last_slot_values'] = _sort_dictionary(data_item['last_slot_values'])
    return_data_item[f'last_pred_{parsing_func}'] = _sort_dictionary(data_item[f'last_pred_{parsing_func}']) if data_item[f'last_pred_{parsing_func}'] else None
    
    return_data_item['pred_prior_context'] = _sort_dictionary(data_item['pred_prior_context'])

    return_data_item['pred'] = _sort_dictionary(data_item['pred'])
    return_data_item['pred_delta_slot_values'] = _sort_dictionary(data_item['pred_delta_slot_values'])

    # return_data_item['prompt'] = data_item['prompt']
    # return_data_item['num_solutions'] = data_item['num_solutions']
    # return_data_item['prompt_counts'] = data_item['prompt_counts']
    # return_data_item['examples'] = data_item['examples']
    # # return_data_item['rights'] = data_item['rights']

    # for key in data_item.keys():
    #     if key not in return_data_item.keys():
    #         return_data_item[key] = data_item[key]

    return return_data_item

def unroll_or(gold, pred):
    for slot, val in gold.items():
            if '|' in val:
                for vv in val.split('|'):
                    if pred.get(slot) == vv:
                        pred[slot] = vv
                        gold[slot] = vv
                        break
    return gold, pred

def update_dialogue_state(context: MultiWOZDict, normalized_turn_parse: MultiWOZDict) -> MultiWOZDict:
    """
    This code is originally FROM REFPYDST
    Given a normalized parse for a turn state and an existing prior state, compute the new complete
    updated state.

    :param context: complete state at turn t - 1
    :param normalized_turn_parse: predicted state change at turn t
    :return: complete state at turn t
    """
    new_dialogue_state: MultiWOZDict = copy.deepcopy(context)
    for slot_name, slot_value in normalized_turn_parse.items():
        if slot_name in new_dialogue_state and slot_value == "[DELETE]":
            del new_dialogue_state[slot_name]
        elif slot_value != "[DELETE]":
            new_dialogue_state[slot_name] = slot_value
    return new_dialogue_state
