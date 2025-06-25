from collections import defaultdict
from typing import Dict, List, Any
from tqdm import tqdm

from evaluate_metrics import evaluate


def evaluate_logs(running_log, test_set, turn=-1) -> Dict[str, Any]:
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    result_dict: Dict[int, List[int]] = defaultdict(list)  # use to record the accuracy

    # start experiment
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0

    for data_item, label_item in tqdm(zip(running_log, test_set)):
        assert data_item['ID'] == label_item['ID'], \
            f"mismatched dialogues: {data_item['ID']}, {label_item['ID']}"
        assert data_item['turn_id'] == label_item['turn_id'], \
            f"mismatched dialogue turns: {data_item['turn_id']}, {label_item['turn_id']}"

        if turn >= 0:
            if data_item['turn_id'] != turn:
                continue

        n_total += 1

        # aggregate the prediction and the history states
        predicted_slot_values = data_item['pred']

        this_jga, this_acc, this_f1 = evaluate(
            predicted_slot_values, label_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
        else:
            result_dict[data_item['turn_id']].append(0)
    jga: float = n_correct / n_total
    slot_acc: float = total_acc / n_total
    joint_f1: float = total_f1 / n_total
    print(f"correct (JGA) {n_correct}/{n_total}  =  {jga:.4f}")
    print(f"Slot Acc {slot_acc:.4f}")
    print(f"Joint F1 {joint_f1:.4f}")
    print()

    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v):.4f}")

    return {
        "slot_acc": slot_acc,
        "joint_f1": joint_f1,
        "jga": jga,
        "turn_accuracies": {k: sum(v) / len(v) for k, v in result_dict.items()}
    }
