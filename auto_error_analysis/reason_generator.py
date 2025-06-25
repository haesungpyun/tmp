from collections import defaultdict
import copy
from vllm import LLM, SamplingParams
import re


class ReasonGenerator(object):
    def __init__(
        self,
        llm_config: dict
    ):  
        self.LLM = LLM(model=llm_config['engine'], quantization=llm_config['quantization'], enforce_eager=True)
        self.tokenizer = self.LLM.get_tokenizer()
        self.terminators =  [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")    
        ]

        self.sampling_params = SamplingParams(
            n=1, best_of=1, max_tokens=300, 
            temperature=0, stop_token_ids=self.terminators)
    
    def word_in_sentence(self, sentence, word, is_domain=False):
        sys_prompt = "**INSTRUCTION**\n\n"
        if is_domain:
            sys_prompt += "Your task is to identify the given **utterance** is about the given **domain**"
        else:
            sys_prompt += "Your task is to identify the given **utterance** is about the given **slot**"
        sys_prompt += "You will be provided with:\n\n"
        sys_prompt += "- **Utterance**: The system or user utterance in the dialogue.\n"
        if is_domain:
            sys_prompt += "- **Domain**: The domain you need to identify in the utterance.\n\n"
        else:
            sys_prompt += "- **Slot**: The slot you need to identify in the utterance.\n\n"
        sys_prompt += "**What to Do**:\n\n"
        sys_prompt += "1. **Read the Utterance**: Carefully read the system or user utterance to understand its content.\n"
        if is_domain:
            sys_prompt += "2. **Identify the Domain**: Determine if the utterance is related to the given domain.\n"
            sys_prompt += "3. **Return Your Answer**: Provide a binary answer indicating whether the utterance is about the domain.\n\n"
        else:
            sys_prompt += "2. **Identify the Slot**: Determine if the utterance contains the given slot.\n"
            sys_prompt += "3. **Return Your Answer**: Provide a binary answer indicating whether the utterance is about the slot.\n\n"
        
        user_prompt = f"**Utterance**:\n"
        user_prompt += f"    {sentence}\n"
        if is_domain:
            user_prompt += f"**Domain**:\n"
            user_prompt += f"    {word}\n"
        else:
            user_prompt += f"**Slot**:\n"
            user_prompt += f"    {word}\n"
        
        user_prompt += f"\n Provide only a single scalar value as output. If the utterance contains the given domain or slot, return 1; otherwise, return 0.\n"
        assistant_prompt = f"**Answer**: \n"

        prompt = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user","content": user_prompt},
                    {"role": "assistant","content": assistant_prompt}
                ]
        prompt_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
        prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)[0]
        
        
        result = self.LLM.generate(prompts, sampling_params=self.sampling_params)
        completions = result[0].outputs[0].text.lower()
        try:
            scalar = re.sub(r"[^0-9 ]", "", completions)
            if len(scalar) > 0:
                scalar = int(scalar)
            else:
                scalar = None
        except Exception as e:
            if 'yes' in completions:
                scalar = 1
            elif 'no' in completions:
                scalar = 0
            else:
                scalar = None
            print(f"Error in generating the reason for the error: {e}")

        return scalar
    
    def identify_action_target(
        self, context, gold_delta_bs, 
        pred_delta_bs, prev_pred_bs, 
        error_name, error_s_v_pairs
    ):
        sys_prompt ="""
**Task Overview:**

You are tasked with analyzing a model's performance in a Dialogue State Tracking (DST) task. Using the provided information, you will identify semantic errors, classify the errors, and infer the root causes from the model's capability perspective. Your analysis should culminate in a **single, clear key phrase** summarizing the likely cause of each error.

---

**Provided Information:**

1. **Model's Input:**
   - **Dialogue History:** A JSON-formatted summary of all previous dialogue states.
   - **Recent Dialogue:** The most recent exchange between the user and the system, upon which the model predicts changes in the dialogue state.

2. **Model's Output:**
   - **Predicted Dialogue State Change:** The model's predicted updates to the dialogue state based on the recent dialogue (in JSON format).

3. **Ground Truth:**
   - **Actual Dialogue State Change:** The true dialogue state changes that should have occurred based on the recent dialogue (in JSON format).

4. **Errors:**
   - **Mechanical Error Type:** The specific type of error identified by simply checking the absence or presence of slots and values pair.
   - **Error Slot and Value:** The slot and value pair that was incorrectly predicted or missed by the model.

---

**Instructions:**

1. **Understand the Dialogue Context:**
   - **Assess the Dialogue History and Recent Dialogue** to comprehend the conversation flow, user intents, and any context shifts.

2. **Identify Semantic Errors:**
   - **Compare the Model's Output with the Ground Truth** to identify discrepancies in slot-value pairs or dialogue state changes.

3. **Infer the Root Cause:**
   - **Answer the Question:** "Why did the error occur?"
   - **Base your inference on:**
     - The **Model's Input** (Dialogue History and Recent Dialogue)
     - The **Model's Output**
     - The **Ground Truth**
     - **Any relevant previous predictions**
   - **Consider aspects such as:**
     - Misinterpretation of the dialogue history
     - Missing user confirmations or updates
     - Failing to detect context changes
     - Over-reliance on previous dialogue states
     - Any other relevant factors affecting the model's performance

4. **Classify Each Error Based on Your Inference:**
   - **Using your inferred root cause**, select the **Incorrect Action** the model performed from the **Action List** below. **DO NOT choose an option that is not in the list.**
   - Identify the **Target** of this action by choosing from the **Target List** below. **DO NOT choose an option that is not in the list.**
   - Ensure that your selections accurately reflect the conclusions drawn in your inference.
   - DO NOT include any additional information beyond the root cause in your response.

   **Action List:**

   - a. 'failed memorizing schema'
   - b. 'incorrectly interpreted'
   - c. 'incorrectly assumed'
   - d. 'mistakenly added'
   - e. 'oversensitively captured'
   - f. 'failed to link'
   - g. 'incorrectly linked'
   - h. 'missed'
   - i. 'incorrectly extracted specific value'
   - j. 'over-relied on previous state'
   - k. 'prematurely captured'
   - l. 'incorrectly overwrote'

   **Target List:**

   - a. 'user's intent'
   - b. 'user's question'
   - c. 'user's confirmation'
   - d. 'user's refusal'
   - e. 'user's flexible preference'
   - f. 'system's suggestion'
   - g. 'system's description'
   - h. 'non-existent slot-value'
   - i. 'topic shift'
   - j. 'information in the recent dialogue with the information in the dialogue history'
   - k. 'two pieces of information in the recent dialogue'

5. **Provide a Concise Explanation:**
   - **Summarize your analysis in one single, clear key phrase** that explains the likely cause of the model's error from its perspective.

---"""

        user_prompt_1 = """

**Example:**

*Given:*

- **Model's Input:**
  - **Dialogue History:**
    {"restaurant-area": "centre", "restaurant-food": "chinese", "restaurant-pricerange": "expensive", "train-arriveby": "14:00", "train-day": "saturday", "train-departure": "cambridge", "train-destination": "stevenage", "train-leaveat": "14:00"}

  - **Recent Dialogue:**
    - **System:** "I have 4 options. HK Fusion is an excellent choice."
    - **User:** "Ok, can I please book a table for 6 people at 11:15 on the same day?"

- **Model's Output (Predicted Dialogue State Change):**
  {"restaurant-book day": "saturday", "restaurant-book people": "6", "restaurant-book time": "11:15"}

- **Ground Truth (Actual Dialogue State Change):**
  {"restaurant-book day": "saturday", "restaurant-book people": "6", "restaurant-book time": "11:15", "restaurant-name": "hk fusion"}

- **Error:**
   - **Mechanical Error Type:** 
   'totally miss slot-value'
   - **Error Slot and Value:** 
   {"restaurant-name": "hk fusion"}

---

**Final Output Format:**

**Analysis:**

1. **Identify Semantic Error:**
2. **Inferred Reason for the Semantic Error:**
3. **Incorrect Action from **Action List**:**
4. **What from **Target List**:**
5. **Concise Explanation:**

"""
        assistant_prompt ="""

**Analysis:**

1. **Identify Semantic Error:**
   The model failed to include the `'restaurant-name': 'hk fusion'` in its predicted dialogue state change.

2. **Inferred Reason for the Semantic Error:**
   The model did not recognize that the user's response implied acceptance of the system's suggestion of "HK Fusion" as the restaurant to book.
   
3. **Incorrect Action from **Action List**:**
   Action h. `'missed'`

4. **What from **Target List**:**
   Target c. `'user's confirmation'`

5. **Concise Explanation:**
   **"Missed the user's implicit confirmation of 'HK Fusion' due to not interpreting acceptance of the system's suggestion."**

---"""


        user_prompt_2 = f"""
**Example:**

*Given:*

- **Model's Input:**
- **Dialogue History:**
    {prev_pred_bs}

- **Recent Dialogue:**
    - **System:** "{context["sys"][-1]}"
    - **User:** "{context["usr"][-1]}"

- **Model's Output (Predicted Dialogue State Change):**
{pred_delta_bs}

- **Ground Truth (Actual Dialogue State Change):**
{gold_delta_bs}

- **Error:**
   - **Mechanical Error Type:** 
   {error_name}
   - **Error Slot and Value:** 
   {error_s_v_pairs}

---"""
        prompt = '<|begin_of_text|>'
        for message, role in zip([sys_prompt, user_prompt_1, assistant_prompt, user_prompt_2], ['system', 'user', 'assistant', 'user']):
            prompt += '<|start_header_id|>'
            prompt += role
            prompt += '<|end_header_id|>\n\n'
            prompt += message
            prompt += '<|eot_id|>'
        # Add the start of an assistant message for the model to complete.
        prompt += '<|start_header_id|>'
        prompt += 'assistant'
        prompt += '<|end_header_id|>\n\n'

        result = self.LLM.generate(prompt, sampling_params=self.sampling_params)

        try:
            completions = result[0].outputs[0].text
            analysis = completions.lower()

            suffix = analysis.split('3.')[1]
            action = suffix.split('4.')[0].strip()
            action = re.sub(r"[^a-zA-Z ]", "", action).strip()
            action = action.replace('incorrect action from action list   action','').strip()

            suffix = analysis.split('4.')[1]
            target = suffix.split('5.')[0].strip()
            target = re.sub(r"[^a-zA-Z ]", "", target).strip()
            target = target.replace('what from target list   target  ','').strip()

            cause = action + " " + target
            # analysis = analysis.replace("**", '')
            return action, target, analysis

        except Exception as e:
            raise Exception(f"Error in generating the reason for the error: {e}")
            return None, None

    def classify_reason(
        self,
        analyzed_log: dict,
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
                    'annotation_error': f"This means that the model made '{error_type }' error because the annotations provided were incorrect in the context of the dialogue",
                    'complete_hallucination': f"This means that the model made '{error_type }' error because it hallucinated {slot_name} and {value_name} which are completely outside the context of the dialogue.",
                    'contextual_hallucination': f"This means that the model made '{error_type }' error because it captures a {slot_name} and {value_name} that are irrelevant to the current dialogue, even though they are mentioned in the dialogue context.",
                    'coreference_error': f"This means the model made '{error_type }' error because it failed to resolve co-references, effectively connecting information from the current utterance with relevant information from the previous dialogue context.",
                    'intent_misunderstanding': f"This means that the model made '{error_type }' error because it misunderstood the user's intent from the utterances.",
                    'incorrect_update': f"This means that the model made '{error_type }' error because  it incorrectly updated the {slot_name} with an incorrect value or assigned the wrong value to a slot.",
                    'late_prediction': f"This means that the model made '{error_type }' error because it predicted slot and value that should have been predicted in earlier turns.",
                    'time_error': f"This means that the model made '{error_type }' error because  it captured an incorrect time value in the dialogue state.",
                }
                
            if 'delta_miss' in error_type:
                slot_name, value_name = error_slot_value[-2], error_slot_value[-1]
                dicts =  {
                    'annotation_error': f"This means that the model made '{error_type }' error because the annotations provided were incorrect in the context of the dialogue",
                    'context_change_miss': f"This means that the model made '{error_type }' error because  it failed to recognize or appropriately handle changes in the dialogue context.",
                    'coreference_error': f"This means the model made '{error_type }' error because it failed to resolve co-references, effectively connecting information from the current utterance with relevant information from the previous dialogue context.",                    
                    'domain_slot_misalignment': f"This means that the model made '{error_type }' error because the {slot_name} and {value_name} were not correctly aligned with the appropriate domain.",
                    'miss_multiple_intents': f"This means that the model made '{error_type }' error because it failed to capture and handle multiple intents from the user's utterances.",
                    'missing_preference': f"This means that the model made '{error_type }' error because  it failed to capture the user's flexible preferences from their utterances.",
                    'slot_value_misalignment': f"This means that the model made '{error_type }' error because the value {value_name} assigned to the slot {slot_name} does not correctly align with the slot's expected type or purpose.",
                    'user_explicit_confirmation_miss': f"This means that the model made '{error_type }' error because it failed to capture the {slot_name} and {value_name} that the user explicitly confirmed.",
                    'user_implicit_confirmation_miss': f"This means that the model made '{error_type }' error because it  it failed to capture the {slot_name} and {value_name} that the user implicitly confirmed.",
                    'user_request_miss': f"This means that the model made '{error_type }' error because it failed to capture the {slot_name} and {value_name} that the user requested.",
                    'user_state_miss': f"This means that the model made '{error_type }' error because it failed to capture the {slot_name} and {value_name} reflecting the user's current state.",
                    'user_refusal_miss': f"This means that the model made '{error_type }' error because it failed to recognize the user's refusal in their utterances.",
                }            
            return dicts
    
        for idx, analyzed_item in enumerate(analyzed_log):
            # 분석된 항목의 정보를 가져옵니다.
            context = analyzed_item.get('dialog')
            gold_delta_bs = analyzed_item.get('turn_slot_values')
            pred_delta_bs = analyzed_item.get(f'pred_delta_{self.parsing_func.__name__}')
            analyzed_item['error_reason'] = copy.deepcopy(analyzed_item.get('error', []))

            # iterate over the error cases
            for i, (error_type, error_slot_value) in enumerate(analyzed_item.get('error', [])):
                
                if 'error_prop' in error_type:
                    analyzed_item['error_reason'][i] = (error_type, error_slot_value, 'error_propagation')
                    continue
                        
                sys_prompt = "You are a talented error analyzer! Let's analyze the error cases and find out the cause of it together. \n\n"            
                sys_prompt += "**INSTRUCTION**\n\n"
                sys_prompt += "You will be provided with a **Dialogue** between a system and a user, along with the **Gold Standard Dialogue State**, the **Predicted Dialogue State Change**, an **Error**, and a **List of Possible Reasons**.\n\n"

                sys_prompt += "Your task is to analyze the **Error** by identifying the most likely cause from the **List of Possible Reasons**. "
                sys_prompt += "Base your analysis on the discrepancies between the **Gold Standard Dialogue State** and the **Predicted Dialogue State Change**, considering the overall context of the **Dialogue**. " 
                sys_prompt += "Return the **Index** of the cause that best explains the **Error** from the **List of Possible Reasons**. \n\n"

                sys_prompt += "**Analysis Criteria**:\n"
                sys_prompt += "1. The cause should be identifiable from the context of the **Dialogue** and the mismatch between the **Gold Standard Dialogue State** and the **Predicted Dialogue State Change**. \n"
                sys_prompt += "2. The cause should consider the **Dialogue**, the **Gold Standard Dialogue State**, and the **Predicted Dialogue State Change** to explain the **Error**. \n"
                sys_prompt += "3. The identified cause should reflect the nature of the discrepancy between the **Gold Standard Dialogue State** and the **Predicted Dialogue State Change**. \n"
                sys_prompt += "4. The cause should logically align with the **Error** and the overall context of the **Dialogue**. \n"
                sys_prompt += "5. The identified cause must match one of the options in the **List of Possible Reasons**. \n\n"

                sys_prompt += "**Analysis Steps**:\n"
                sys_prompt += "1. **Examine the Dialogue**: Review the conversation to understand the context and the system's behavior. \n"
                sys_prompt += "2. **Compare State Changes**: Analyze the **Gold Standard Dialogue State** and the **Predicted Dialogue State Change** to identify discrepancies. \n"
                sys_prompt += "3. **Assess the Error**: Validate that the identified discrepancy corresponds to the provided **Error**. \n"
                sys_prompt += "4. **Evaluate Possible Reasons**: Analyze the **List of Possible Reasons** and identify which one best explains the **Error** based on the **Dialogue**, **Gold Standard Dialogue State**, and **Predicted Dialogue State Change**. \n"
                sys_prompt += "5. **Return the Cause Index**: Provide the index of the most appropriate cause from the **List of Possible Reasons**, without additional commentary. \n\n"
                
                user_prompt = f"**Dialog**:\n"
                user_prompt += f"{analyzed_item['last_slot_values']}\n"
                # for sys_utt, usr_utt in zip(context['sys'], context['usr']):
                user_prompt += f" System: {context['sys'][-1]}\n"
                user_prompt += f" User: {context['usr'][-1]}\n"

                user_prompt += f"\n**Gold Standard Dialogue State Change**:\n"
                user_prompt += f"    {gold_delta_bs}\n\n"
                user_prompt += f"**Predicted Dialogue State Change**:\n"
                user_prompt += f"    {pred_delta_bs}\n\n"

                user_prompt += f"**Error**: \n"
                user_prompt += f"    {error_type}: " + reason_dict[error_type]

                explanation_list = list(retrun_explanation_dict(error_type, error_slot_value).items())

                user_prompt += "\n\n**List of Possible Reasons**: \n\n"
                for idx,(reason_name, explanaton) in enumerate(explanation_list):
                    user_prompt += f"({idx}) **{reason_name}**\n"
                    user_prompt += f"{explanaton} \n\n"

                # user_prompt += f"Provide only a single scalar value as output.\n"
                user_prompt += f"**Index**: \n"

                prompt = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user","content": user_prompt}
                ]

                # Make a request to the LLM to generate the reason for the error.
                prompt_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
                prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)[0]
                try:
                    result = self.LLM.generate(prompts, sampling_params=self.sampling_params)

                    completions = result[0].outputs[0].text
                    scalar = re.sub(r"[^0-9 ]", "", completions)
                    scalar = int(scalar)

                    reason = explanation_list[scalar][0]
                except Exception as e:
                    reason = None
                    print(f"Error in generating the reason for the error: {e}")

                analyzed_item['error_reason'][i] = (error_type, error_slot_value, reason)

        return analyzed_item



    def generate_reason(
        self, context, gold_delta_bs, 
        pred_delta_bs, prev_pred_bs, 
        error_name, error_s_v_pairs
    ):

        reason_dict = {
            "delta_hall_total": "This error is occred because the model hallucinated the slot and value. ",
            "delta_hall_overwrite": "This error is occred because the model updated the slot and value with incorrect value. ",
            "delta_hall_val": "This error is occred because the model correctly captures slot but hallucinated the value. ",
            "delta_miss_total": "This error is occred because the model missed the slot and value in the dialog. ",            
            "delta_miss_confuse": "This error is occred because the model confused the slot with another slot. ",
            "delta_miss_delete": "This error is occred because the model missed the slot and value which was deleted. ",
            "delta_miss_dontcare": "This error is occred because the model missed the user's flexible preference. ",
        }

        sys_prompt = "You are a talented error analyzer! Let's find out the cause of errors together. \n\n"            
        sys_prompt += "**INSTRUCTION**\n\n Your task is to identify the fundamental reason why the model made the given **Error** using **a single key phrase** focusing on the dialog context and current utterances.\n\n"
        sys_prompt += "You will be provided with:\n\n"
        sys_prompt += "- **Model Input**: Conversation history in JSON format and the most recent system-user dialogue in text.\n"
        sys_prompt += "- **Predicted Dialogue State Change**: The model's predicted dialogue state changes for the most recent dialogue.\n"
        sys_prompt += "- **Ground Truth Dialogue State Change**: The correct dialogue state changes for the most recent dialogue.\n\n"
        sys_prompt += "**What to Do**:\n\n"
        sys_prompt += "1. **Compare** the **Predicted Dialogue State Change** with the **Ground Truth Dialogue State Change** to find discrepancies in intents, slots, or slot values.\n"
        sys_prompt += "2. **Consider** the conversation context and dialogue history to understand potential misinterpretations or misunderstandings by the model.\n"
        sys_prompt += "3. **Determine** the fundamental reason for the **Error** based on your analysis.\n\n"
        sys_prompt += "**Your Response**:\n\n"
        sys_prompt += "- Provide **a single key phrase** that best explain the reason for the model's error.\n"
        sys_prompt += "- Do **not** include any additional text or explanation.\n\n"
        sys_prompt += "**Analysis Criteria for Error Cases**:\n\n"
        sys_prompt += "1. **Discrepancy Identification**:\n"
        sys_prompt += "   - **Description:** Identify the specific differences between the **Predicted Dialogue State Change** and the **Ground Truth Dialogue State Change**. These discrepancies are the starting point for understanding the error.\n"
        sys_prompt += "   - **Considerations:**\n"
        sys_prompt += "     - **Missing or Extra Keys:** Are there any intents or slots present in one but not the other?\n"
        sys_prompt += "     - **Incorrect Slot Values:** Do any slots have differing values between the predicted and ground truth states?\n"
        sys_prompt += "     - **Structural Differences:** Is there a mismatch in the format or structure of the dialogue state changes?\n\n"
        sys_prompt += "2. **Contextual Understanding and Interpretation**:\n"
        sys_prompt += "   - **Description:** Assess whether the model accurately understood the conversation history and the most recent system-user dialogue. Misinterpretation can lead to errors in state changes.\n"
        sys_prompt += "   - **Considerations:**\n"
        sys_prompt += "     - **Relevance of History:** Did the model utilize pertinent parts of the conversation history to inform its prediction?\n"
        sys_prompt += "     - **Ambiguity Handling:** How did the model interpret ambiguous or implicit information within the dialogue?\n"
        sys_prompt += "     - **Temporal Dependencies:** Did the model correctly associate references to previous intents or entities?\n\n"
        sys_prompt += "3. **Intent Recognition and Slot Filling Accuracy**:\n"
        sys_prompt += "   - **Description:** Evaluate the model's ability to correctly identify user intents and accurately extract and assign slot values, which are crucial for correct dialogue state updates.\n"
        sys_prompt += "   - **Considerations:**\n"
        sys_prompt += "     - **Intent Misclassification:** Was the user's intent correctly identified based on the input dialogue?\n"
        sys_prompt += "     - **Slot Extraction Errors:** Were all relevant slots accurately identified and filled with correct values?\n"
        sys_prompt += "     - **Multiple Slot Handling:** Did the model effectively manage dialogues requiring updates to multiple slots simultaneously?\n\n"
        sys_prompt += "4. **Logical Consistency and Reasoning**:\n"
        sys_prompt += "   - **Description:** Determine if the model's predictions are logically consistent with the conversation and if it applied appropriate reasoning.\n"
        sys_prompt += "   - **Considerations:**\n"
        sys_prompt += "     - **Flow Consistency:** Does the predicted state logically follow from the conversation history and the latest dialogue?\n"
        sys_prompt += "     - **Rule Adherence:** Did the model comply with any predefined rules or constraints governing the dialogue?\n"
        sys_prompt += "     - **Inference Capabilities:** Was the model able to infer necessary information that was not explicitly stated?\n\n"
        sys_prompt += "5. **Error Summarization with key phrase**:\n"
        sys_prompt += "   - **Description:** Condense the fundamental reason for the error into a single representative key phrase that encapsulate the core issue.\n"
        sys_prompt += "   - **Considerations:**\n"
        sys_prompt += "     - **Core Issue Identification:** Pinpoint the primary factors that led to the error.\n"
        sys_prompt += "     - **Key Word Selection:** Choose key phrase that are specific, relevant, and collectively provide a clear picture of the error's root cause.\n"
        sys_prompt += "     - **Clarity and Relevance:** Ensure the selected key phrase accurately reflect the analysis and are easily understood without additional context.\n\n"
        sys_prompt += "**Analysis Steps:**\n\n"
        sys_prompt += "1. **Understand the Context**:\n"
        sys_prompt += "   - **Read the Model Input**: Carefully read the conversation history in JSON format and the most recent system-user dialogue in text to fully grasp the context, nuances, and flow of the dialogue.\n\n"
        sys_prompt += "2. **Review Dialogue State Changes**:\n"
        sys_prompt += "   - **Examine Predicted State Change**: Review the **Predicted Dialogue State Change** to understand what the model has outputted for the latest dialogue.\n"
        sys_prompt += "   - **Examine Ground Truth State Change**: Review the **Ground Truth Dialogue State Change** to know the correct dialogue state changes expected.\n\n"
        sys_prompt += "3. **Identify Discrepancies**:\n"
        sys_prompt += "   - **Compare State Changes**: Systematically compare the predicted and ground truth dialogue state changes to identify discrepancies in intents, slots, or slot values.\n"
        sys_prompt += "   - **Document Differences**: Note any missing or extra intents or slots, incorrect slot values, or structural mismatches between the two state changes.\n\n"
        sys_prompt += "4. **Analyze Potential Misinterpretations**:\n"
        sys_prompt += "   - **Contextual Understanding**: Assess if the model may have misunderstood the conversation context or failed to leverage relevant parts of the dialogue history.\n"
        sys_prompt += "   - **Ambiguity Handling**: Consider whether ambiguous or implicit information in the dialogue led to misinterpretation.\n"
        sys_prompt += "   - **Temporal Dependencies**: Check if the model failed to correctly associate references to previous intents or entities.\n\n"
        sys_prompt += "5. **Evaluate Intent Recognition and Slot Filling**:\n"
        sys_prompt += "   - **Intent Classification**: Determine if the model correctly identified the user's intent based on the input dialogue.\n"
        sys_prompt += "   - **Slot Extraction**: Evaluate whether all relevant slots were accurately identified and filled with correct values.\n"
        sys_prompt += "   - **Multiple Slot Handling**: Assess the model's ability to handle dialogues requiring updates to multiple slots simultaneously.\n\n"
        sys_prompt += "6. **Assess Logical Consistency and Reasoning**:\n"
        sys_prompt += "   - **Flow Consistency**: Verify if the predicted state logically follows from the conversation history and latest dialogue.\n"
        sys_prompt += "   - **Rule Adherence**: Confirm that the model adhered to any predefined rules or constraints governing the dialogue.\n"
        sys_prompt += "   - **Inference Capability**: Determine if the model made necessary inferences when explicit information was not provided.\n\n"
        sys_prompt += "7. **Determine the Fundamental Reason for the Error**:\n"
        sys_prompt += "   - **Synthesize Analysis**: Based on the discrepancies and your assessments, identify the primary factors that led to the model's error.\n"
        sys_prompt += "   - **Focus on Key Factors**: Pinpoint the most significant issues contributing to the error.\n\n"
        sys_prompt += "8. **Select A Single key phrase**:\n"
        sys_prompt += "   - **Summarize the Error**: Condense the fundamental reason into a single representative key phrase that encapsulate the core issue.\n"
        sys_prompt += "   - **Ensure Clarity and Relevance**: Choose key phrase that are specific, relevant, and collectively provide a clear picture of the root cause.\n\n"
        sys_prompt += "9. **Provide Your Response**:\n"
        sys_prompt += "   - **List the A Single key phrase**: Present the a single key phrase as your final answer.\n"
        sys_prompt += "   - **Follow Instructions**: Do not include any additional text or explanation beyond the a single key phrase."


        if 'error_prop' in error_name:            
            return 'error_propagation'
        
        user_prompt = f"**Model Input**:\n"
        user_prompt += f"  {prev_pred_bs}\n"
        user_prompt += f"   System: {context['sys'][-1]}\n"
        user_prompt += f"   User: {context['usr'][-1]}\n\n"

        user_prompt += f"**Predicted Dialogue State Change**:\n"
        user_prompt += f"    {pred_delta_bs}\n\n"

        user_prompt += f"**Gold Standard Dialogue State Change**:\n"
        user_prompt += f"    {gold_delta_bs}\n\n"

        user_prompt += f"**Error**: \n"
        user_prompt += f"    {reason_dict[error_name]} \n"

        if 'miss' in error_name:
            user_prompt += f"Why the model failed to correctly extract the slot-value pair in the dialogue state change? \n\n"
        else:
            user_prompt += f"Why the model hallucinated the {error_s_v_pairs[-2]} and {error_s_v_pairs[-1]} in the dialogue state change? \n\n"
        
        assistant_prompt = "**Fundatmetal Reason**: \n"
    
        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user","content": user_prompt},
            {"role": "assistant","content": assistant_prompt}
        ]

        # Make a request to the LLM to generate the reason for the error.
        prompt_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
        prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)[0]
        
        result = self.LLM.generate(prompts, sampling_params=self.sampling_params)

        completions = result[0].outputs[0].text
        reason = completions.replace("**Root Cause**:", '').strip()
        
        return reason
