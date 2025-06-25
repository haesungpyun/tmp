"""
FROM REFPYDST
This file was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as originally
published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

@article{hu2022context,
  title={In-Context Learning for Few-Shot Dialogue State Tracking},
  author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2203.08568},
  year={2022}
}
"""
import enum
from collections import namedtuple, Counter
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
from data_types import MultiWOZDict, Turn

EvalResult = namedtuple('EvalResult', ['jga', 'acc', 'f1'])
PRFResult = namedtuple('PRFResult', ['f1', 'precision', 'recall'])

# counts of informable slots for each domain
INFORMABLE_SLOTS_BY_DOMAIN: Dict[str, int] = {
    "attraction": 3,
    "hotel": 10,
    "restaurant": 7,
    "taxi": 4,
    "train": 6
}

# inheriting from str + enum.Enum allows for painless JSON serialization
class F1Class(str, enum.Enum):
    TP = 'tp'
    FP = 'fp'
    FN = 'fn'


def calc_prf(counter: Counter[F1Class]) -> PRFResult:
    precision_denom: int = (counter[F1Class.TP] + counter[F1Class.FP])
    precision: float = counter[F1Class.TP] / precision_denom if precision_denom else 0
    recall_denom: int = (counter[F1Class.TP] + counter[F1Class.FN])
    recall: float = counter[F1Class.TP] / recall_denom if recall_denom else 0
    f1: float = 0
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    return PRFResult(f1, precision, recall)


def compute_prf(gold: MultiWOZDict, pred: MultiWOZDict) -> PRFResult:
    """
    Compute the f1, precision, and recall for a turn-level prediction, given normalized MultiWOZ slots.
    :param pred: flattened and normalized predicted slots, e.g. {"hotel-area": "centre", ...}
    :param gold: flattened and normalized gold reference slots, e.g. {"hotel-area": "centre", ...}
    :return: named tuple (f1, precision, recall)
    """
    scores: Counter[F1Class] = Counter()
    for gold_slot, gold_value in gold.items():
        # if the slot is present in prediction AND the values match -> TP else FN
        if gold_slot in pred and pred[gold_slot] == gold_value:
            scores[F1Class.TP] += 1
        else:
            scores[F1Class.FN] += 1
    for pred_slot, pred_value in pred.items():
        if pred_slot not in gold:
            scores[F1Class.FP] += 1
    return calc_prf(counter=scores)


def compute_acc(gold: MultiWOZDict, pred: MultiWOZDict, number_of_informable_slots: int = 30) -> float:
    """
    Compute the accuracy of a turn-level prediction, given normalized MultiWOZ slots. The number of total informable
    slots must be known, so that correctly absent entries in both the prediction and gold reference can be properly
    counted as accurate.

    :param pred: flattened and normalized predicted slots, e.g. {"hotel-area": "centre", ...}
    :param gold: flattened and normalized gold reference slots, e.g. {"hotel-area": "centre", ...}
    :param number_of_informable_slots: number of informable slots, 30 in IC-DST experiments
    :return: slot accuracy on this turn
    """
    false_negatives: int = 0
    missed_slots: List[str] = []
    # count up false negatives
    for domain_and_slot_name in gold:
        if domain_and_slot_name not in pred or gold[domain_and_slot_name] != pred[domain_and_slot_name]:
            false_negatives += 1
            # get just the slot name
            missed_slots.append(domain_and_slot_name.rsplit("-", 1)[0])
    false_positives = 0
    for domain_and_slot_name in pred:
        if domain_and_slot_name not in gold:
            false_positives += 1
    acc: float = (number_of_informable_slots - false_negatives - false_positives) / number_of_informable_slots
    return acc


def evaluate(prediction: MultiWOZDict, gold: MultiWOZDict) -> EvalResult:
    """
    Evaluates a single prediction against a gold reference, each in standardized MultiWOZ format.

    :param prediction: flattened and normalized predicted slots, e.g. {"hotel-area": "centre", ...}
    :param gold: flattened and normalized gold reference slots, e.g. {"hotel-area": "centre", ...}
    :return: a named tuple with joint-goal accuracy, slot accuracy, and slot F1
    """

    for key in gold.keys():
        # if the gold value supports multiple ground truth values, and we predicted one, set the single-gold value to
        # the one we predicted.
        if '|' in gold[key]:
            gold_values = gold[key].split('|')
            if key in prediction and prediction[key] in gold_values:
                gold[key] = prediction[key]

    # joint-goal can be computed with dict match
    jga: int = 1 if prediction == gold else 0

    acc = compute_acc(gold, prediction)
    f1 = compute_prf(gold, prediction)[0]
    return jga, acc, f1


def evaluate_on_domain(prediction: MultiWOZDict, gold: MultiWOZDict, domain: str) -> EvalResult:
    """
    Evaluates a single prediction against a gold reference on a particular domain, each in standardized
    MultiWOZ format. One does not need to pre-filter out the informable slots for other domains.

    :param prediction: a prediction for a turn, in standard MultiWOZ form
    :param gold: a label for a turn, in standard MultiWOZ form
    :param domain: an evaluation MultiWOZ domain (attraction, hotel, taxi, train, restaurant)
    :return: EvalResult:
       - jga indicates whether all informable slots for the domain are exactly correct
       - acc provides the accuracy among informable slots for that domain
       - f1 provides the F1 among informable slots for that domain
    """
    # first, filter to only slots for this domain
    prediction = {k: v for k, v in prediction.items() if k.split("-")[0] == domain}
    gold = {k: v for k, v in gold.items() if k.split("-")[0] == domain}

    for key in gold.keys():
        # if the gold value supports multiple ground truth values, and we predicted one, set the single-gold value to
        # the one we predicted.
        if '|' in gold[key]:
            gold_values = gold[key].split('|')
            if key in prediction and prediction[key] in gold_values:
                gold[key] = prediction[key]

    # joint-goal can be computed with dict match
    jga: int = 1 if prediction == gold else 0

    acc = compute_acc(gold, prediction, number_of_informable_slots=INFORMABLE_SLOTS_BY_DOMAIN[domain])
    f1 = compute_prf(gold, prediction, )[0]
    return jga, acc, f1

def calculate_token_f1(encoding, gold, pred):
    pred_tokens = encoding.encode(pred)
    gold_tokens = encoding.encode(gold)

    # Token level f1 score
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for token in gold_tokens:
        if token in pred_tokens:
            true_positive += 1
        else:
            false_negative += 1

    for token in gold_tokens:
        if token not in gold_tokens:
            false_positive += 1

    # Calulate f1 avoiding division by zero
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1


def calc_f1(counter: Counter[F1Class]) -> float:
    precision: float = calc_precision(counter)
    recall: float = calc_recall(counter)
    if precision and recall:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


def calc_recall(counter: Counter[F1Class]) -> float:
    recall_denom: int = (counter[F1Class.TP] + counter[F1Class.FN])
    recall: float = counter[F1Class.TP] / recall_denom if recall_denom else 0
    return recall


def calc_precision(counter: Counter[F1Class]) -> float:
    precision_denom: int = (counter[F1Class.TP] + counter[F1Class.FP])
    precision: float = counter[F1Class.TP] / precision_denom if precision_denom else 0
    return precision


def slot_level_f1(logs: List[Turn], tp_means_correct: bool = True, gold_key: str = 'slot_values') -> Dict[
    str, Tuple[Counter[F1Class], float]]:
    slot_scores: Dict[str, Counter[F1Class]] = defaultdict(Counter)
    for turn in tqdm(logs, desc="calculating slot-level F1", total=len(logs)):
        for gold_slot, gold_value in turn[gold_key].items():
            # if the slot is present in prediction, whether its a TP or FN depends on our tp_means_correct flag
            if gold_slot in turn['pred'] and (not tp_means_correct or turn['pred'][gold_slot] == gold_value):
                slot_scores[gold_slot][F1Class.TP] += 1
            else:
                slot_scores[gold_slot][F1Class.FN] += 1
        for pred_slot, pred_value in turn['pred'].items():
            if pred_slot not in turn[gold_key]:
                slot_scores[pred_slot][F1Class.FP] += 1
    return {k: (v, calc_f1(v)) for k, v in slot_scores.items()}