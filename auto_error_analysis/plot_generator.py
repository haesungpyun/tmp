import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict
from tqdm import tqdm

from evaluate_run_log import evaluate_logs
from evaluate_metrics import evaluate, slot_level_f1
from utils import load_analyzed_log



class PlotGenerator:
    def __init__(self, output_dir_path):
        self.output_dir_path = output_dir_path

        self.slot_order = ["attraction-area", "attraction-name", "attraction-type", "bus-day", "bus-departure",
                   "bus-destination", "bus-leaveat", "hospital-department", "hotel-area", "hotel-book day",
                   "hotel-book people", "hotel-book stay", "hotel-internet", "hotel-name", "hotel-parking",
                   "hotel-pricerange", "hotel-stars", "hotel-type", "restaurant-area", "restaurant-book day",
                   "restaurant-book people", "restaurant-book time", "restaurant-food", "restaurant-name",
                   "restaurant-pricerange", "taxi-arriveby", "taxi-departure", "taxi-destination", "taxi-leaveat",
                   "train-arriveby", "train-book people", "train-day", "train-departure", "train-destination",
                   "train-leaveat"]

        self.order_error = [
            'delta_hall_total', 'delta_hall_overwrite', 'delta_hall_val', 
            'delta_miss_total', 'delta_miss_confuse', 'delta_miss_delete', 'delta_miss_dontcare',
            'error_prop_hall_total', 'error_prop_hall_overwrite', 'error_prop_hall_val',
            'error_prop_miss_total', 'error_prop_miss_confuse', 'error_prop_miss_delete', 'error_prop_miss_dontcare', 'format error'
        ]

    def plot_all_stats(self, analyzed_log):
        item = analyzed_log[0]
        self.pred_key = None
        for key in item:
            if key.startswith('pred_delta_'):
                self.pred_key = key
                break

        # Count the error statistics
        if not os.path.exists(self.output_dir_path + '/plots/stats'):
            os.makedirs(self.output_dir_path + '/plots/stats')
        
        if not os.path.exists(self.output_dir_path + '/plots/turn_stats'):
            os.makedirs(self.output_dir_path + '/plots/turn_stats')

        if not os.path.exists(self.output_dir_path + '/plots/detail_stats'):
            os.makedirs(self.output_dir_path + '/plots/detail_stats')

        self.show_stats(analyzed_log)
        
        self.plot_error_stats(analyzed_log)
        self.plot_reason_stats(analyzed_log)
        self.plot_error_turn_stats(analyzed_log)
        self.plot_reason_trun_stats(analyzed_log)
        self.plot_reson_by_error(analyzed_log)
        self.plot_slot_stats(analyzed_log, self.output_dir_path)

        self.show_state_change_confusion_matrix(analyzed_log)


    def plot(self, dicts, name, vertical=True):
        sns.set_theme(style="whitegrid")
        if vertical:
            plt.figure(figsize=(20, 15))
            sns.barplot(x=list(dicts.keys()), y=list(dicts.values()), palette='viridis')
            plt.ylabel('Number of Errors')
            plt.xlabel('Error Type')
            plt.grid(axis='x')
            plt.xticks(rotation=45)
        else:
            plt.figure(figsize=(50, 20))
            sns.barplot(x=list(dicts.values()), y=list(dicts.keys()), palette='viridis')
            plt.ylabel('Number of Errors')
            plt.xlabel('Error Type')
            plt.grid(axis='y')
            plt.yticks(rotation=45)
        plt.title(f'{name} Error Statistics')
            
        for index, (key, value) in enumerate(dicts.items()):
            # plt.text(index, value, (str(value), str(round(value/len(analyzed_log),4))))
            if vertical:
                plt.text(index, value, str(value))
            else:
                plt.text(value, index, str(value))
        
        # save the plot
        plt.savefig(self.output_dir_path + f'{name}.png')

    def plot_error_turn_stats(self, analyzed_log):
        error_turn = defaultdict(int)
        for data_item in analyzed_log:
            tmp_error = defaultdict(int)

            for error_name, error_s_v_pairs in data_item.get('error', [None,None]):
                if error_name is None:
                    continue
                tmp_error[error_name] = 1

            for key in tmp_error.keys():
                error_turn[key] += tmp_error[key]
        
        # Plot the error statistics wiht exact numbers, Sort error_turn by the order of order_error
        error_turn = {k: error_turn.get(k,0) for k in self.order_error}
        if error_turn:
            self.plot(error_turn, '/plots/turn_stats/error_turn_stats', vertical=True)
            self.plot({k: v/sum(error_turn.values()) for k, v in error_turn.items()}, 
                      '/plots/turn_stats/error_turn_stats_ratio', vertical=True)

    def plot_reason_trun_stats(self, analyzed_log):
        reason_turn, action_turn, target_turn= defaultdict(int), defaultdict(int), defaultdict(int)
        for data_item in analyzed_log:
            tmp_reason, tmp_action, tmp_target = defaultdict(int), defaultdict(int), defaultdict(int)

            error_reason = data_item.get('error_reason', [])
            for tuples in error_reason:
                error_name = tuples[0]
                if len(tuples) < 5:
                    reason = tuples[-1]
                else:
                    reason = tuples[3]
                if 'error_prop' in reason or 'raw' in reason or 'format' in reason or 'execution' in reason:
                    continue
                tmp_reason[' '.join(reason)] = 1
                tmp_action[reason[0]] = 1
                tmp_target[reason[1]] = 1

            for key in tmp_reason.keys():
                reason_turn[key] += tmp_reason[key]
            for key in tmp_action.keys():
                action_turn[key] += tmp_action[key]
            for key in tmp_target.keys():
                target_turn[key] += tmp_target[key]
        
        reason_turn = dict(sorted(reason_turn.items(), key=lambda x: x[1], reverse=True))
        action_turn = dict(sorted(action_turn.items(), key=lambda x: x[1], reverse=True))
        target_turn = dict(sorted(target_turn.items(), key=lambda x: x[1], reverse=True))
        if reason_turn:
            self.plot(reason_turn, '/plots/turn_stats/reason_trun_stats', vertical=False)
            print(sum(reason_turn.values()))
            self.plot({k: v/sum(reason_turn.values()) for k, v in reason_turn.items()}, 
                      '/plots/turn_stats/reason_trun_stats_ratio', vertical=False)
        if action_turn:
            self.plot(action_turn, '/plots/turn_stats/action_trun_stats', vertical=False)
        if target_turn:
            self.plot(target_turn, '/plots/turn_stats/target_trun_stats', vertical=False)

    def plot_error_stats(self, analyzed_log):
        error_stats = defaultdict(int)
        for data_item in analyzed_log:
            for error_name, error_s_v_pairs in data_item.get('error', [None,None]):
                if error_name is None:
                    continue
                error_stats[error_name] += 1

        # Plot the error statistics wiht exact numbers
        error_stats = {k: error_stats.get(k, 0) for k in self.order_error}
        
        if error_stats:
            self.plot(error_stats, '/plots/stats/error_stats', vertical=True)

    def plot_reason_stats(self, analyzed_log):
        reason_stats, action_stats, target_stats = defaultdict(int), defaultdict(int), defaultdict(int)
        for data_item in analyzed_log:
            error_reason = data_item.get('error_reason', [])
            for tuples in error_reason:
                if len(tuples) < 5:
                    reason = tuples[-1]
                else:
                    reason = tuples[3]
                if 'error_prop' in reason or 'raw' in reason or 'format' in reason or 'execution' in reason:
                    continue
                reason_stats[' '.join(reason)] += 1
                action_stats[reason[0]] += 1
                target_stats[reason[1]] += 1
                
        # Plot the error statistics wiht exact numbers
        if reason_stats:
            reason_stats = dict(sorted(reason_stats.items(), key=lambda x: x[1], reverse=True))
            self.plot(reason_stats, '/plots/stats/reason_stats', vertical=False)
        if action_stats:
            action_stats = dict(sorted(action_stats.items(), key=lambda x: x[1], reverse=True))
            self.plot(action_stats, '/plots/stats/action_stats', vertical=False)
        if target_stats:
            target_stats = dict(sorted(target_stats.items(), key=lambda x: x[1], reverse=True))
            self.plot(target_stats, '/plots/stats/target_stats', vertical=False)
    
    def plot_reson_by_error(self, analyzed_log):
        reason_by_error, action_by_error, target_by_error = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        for data_item in analyzed_log:
            error_reason = data_item.get('error_reason', [])
            for tuples in error_reason:
                error_name = tuples[0]
                if len(tuples) < 5:
                    reason = tuples[-1]
                else:
                    reason = tuples[3]
                if 'error_prop' in reason or 'raw' in reason or 'format' in reason or 'execution' in reason:
                    continue
                reason_full = ' '.join(reason)
                reason_by_error[error_name][reason_full] = reason_by_error[error_name].get(reason_full, 0) + 1
                action_by_error[error_name][reason[0]] = action_by_error[error_name].get(reason[0], 0) + 1
                target_by_error[error_name][reason[1]] = target_by_error[error_name].get(reason[1], 0) + 1

        for error_name, reason_stats in reason_by_error.items():
            if 'error_prop' in error_name or 'raw' in error_name or 'format' in error_name or 'execution' in error_name:
                continue
            reason_stats = dict(sorted(reason_stats.items(), key=lambda x: x[1], reverse=True))
            self.plot(reason_stats, f'/plots/detail_stats/{error_name}_reason_stats', vertical=False)
        
        for error_name, action_stats in action_by_error.items():
            if 'error_prop' in error_name or 'raw' in error_name or 'format' in error_name or 'execution' in error_name:
                continue
            action_stats = dict(sorted(action_stats.items(), key=lambda x: x[1], reverse=True))
            self.plot(action_stats, f'/plots/detail_stats/{error_name}_action_stats', vertical=False)
        
        for error_name, target_stats in target_by_error.items():
            if 'error_prop' in error_name or 'raw' in error_name or 'format' in error_name or 'execution' in error_name:
                continue
            target_stats = dict(sorted(target_stats.items(), key=lambda x: x[1], reverse=True))
            self.plot(target_stats, f'/plots/detail_stats/{error_name}_target_stats', vertical=False)
          

    def plot_slot_stats(self, analyzed_log, root):
        error_slot_dict = defaultdict(int)
        for error_case in self.order_error:
            slots = defaultdict(int)
            for data_item in analyzed_log:
                error_name, error_s_v = '', ''
                if data_item.get('error') != []:
                    for error in data_item.get('error'):
                        error_name = error[0]
                        error_s_v = error[1]
                        if error_case in error_name:
                            if error_s_v is None:
                                continue    
                            slots[error_s_v[0]] += 1
                            if len(error_s_v) > 2:
                                slots[error_s_v[2]] += 1
            error_slot_dict[error_case] = slots
        
        for error_name, slot_stats in error_slot_dict.items():
            if 'error_prop' in error_name or 'raw' in error_name or 'format' in error_name or 'execution' in error_name:
                continue
            
            slot_stats = {k: slot_stats.get(k, 0) for k in self.slot_order}
            self.plot(slot_stats, f'/plots/detail_stats/{error_name}_slot_stats', vertical=True)
                 
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
        slots = self.slot_order
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
            
            pred_delta_bs = data_item[self.pred_key] if data_item[self.pred_key] else {}
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
                    try:
                        pred_delta_confusion_matrix['hall_value'][pred_slot] += 1
                    except:
                        pred_delta_confusion_matrix['text_hallucination'] += 1

                else:
                    # hall_total: pred: {wrong_slot: wrong_value}
                    try:
                        pred_delta_confusion_matrix['hall_total'][pred_slot] += 1
                    except:
                        pred_delta_confusion_matrix['text_hallucination'] += 1

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
                    try:
                        pred_delta_confusion_matrix['hall_total'][pred_slot] += 1
                    except:
                        pred_delta_confusion_matrix['text_hallucination'] += 1
            else:
                if pred_value == gold_delta_bs[pred_slot]:
                    try:
                        pred_delta_confusion_matrix[pred_slot][pred_slot] += 1
                    except:
                        pred_delta_confusion_matrix['text_hallucination'] += 1
                else:
                    try:
                        pred_delta_confusion_matrix['hall_value'][pred_slot] += 1
                    except:
                        pred_delta_confusion_matrix['text_hallucination'] += 1
                
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

    def show_state_change_confusion_matrix(self, analyzed_log):
        pred_delta_conf_mat, gold_delta_conf_mat = self.make_confusion_matrix(analyzed_log, criteria='value')

        slots = self.slot_order
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

        print('\n=============== Confusion matrix saved to the output directory ===============\n')
        print('PRED Confusion matrix')
        print("# of hal value (Right SLOT Wrong VALUE): ", pred_delta_df.loc['hall_value'].sum())
        print("# of confused (Wrong SLOT Right VALUE): ", pred_delta_df.loc[slots].values.sum() - pred_delta_df.loc[slots].values.diagonal().sum())
        print("# of hall total (Wrong SLOT Wrong VALUE): ", pred_delta_df.loc['hall_total'].sum())
        print()
        print('GOLD Confusion matrix')
        print("# of miss value (Right SLOT Wrong VALUE): ", gold_delta_df.loc[:, 'miss_value'].sum())
        print("# of confused (Wrong SLOT Right VALUE): ", gold_delta_df.loc[slots].values.sum() - gold_delta_df.loc[slots].values.diagonal().sum())
        print("# of miss total (Wrong SLOT Wrong VALUE): ", gold_delta_df.loc[:, 'miss_total'].sum())
        
        plt.savefig(self.output_dir_path + '/plots/confusion_matrix.png')
        return None
    

if __name__ == '__main__':
    import sys

    if os.path.exists(sys.argv[1]):
        dir_path = sys.argv[1]
    # dir_path = '/home/haesungpyun/my_refpydst/final/upperbound/70B/bm_sbert/bm_d_cb_gds_sbert_gds/multiply_div_top_k/split_v1'
    analyzed_log = load_analyzed_log(dir_path)
    plot_generator = PlotGenerator(dir_path)
    plot_generator.plot_all_stats(analyzed_log)

