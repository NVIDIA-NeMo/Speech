# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import namedtuple
from tqdm import tqdm

from nemo.utils import logging

try:
    import pandas as pd
    from tabulate import tabulate

    HAVE_TABLUATE_AND_PANDAS = True
except (ImportError, ModuleNotFoundError):
    HAVE_TABLUATE_AND_PANDAS = False


def punctuation_error_rate(
    references: list[str], hypotheses: list[str], punctuation_marks: list[str], punctuation_mask: str = "[PUNCT]",
) -> None:

    """
    Computes Punctuation Error Rate
    
    Args:
        references (list[str]) - list of references
        hypotheses (list[str]) - list of hypotheses
        punctuation_marks (list[str]) - list of punctuation marks for computing metrics
        punctuation_mask (str, by default "[PUNCT]") - mask token that will be applied to
        given punctuation marks while edit distance calculation
        
    Return:
        punct_er (float) - Punctuation Error Rate
    """

    dper_obj = DatasetPunctuationErrorRate(
        references=references,
        hypotheses=hypotheses,
        punctuation_marks=punctuation_marks,
        punctuation_mask=punctuation_mask,
    )

    dper_obj.compute()

    return dper_obj.punct_er


class OccurancePunctuationErrorRate:
    """
    Class for computation puncutation-related absolute amounts of operations and thier rates
    between reference and hypothesis strings:
        - Absolute amounts of correct predictions, deletions, insertions
        and substitutions for each given punctuation mark
        - Rates of correct predictions, deletions, insertions
        and substitutions for each given punctuation mark
        - Overall rates of correct predictions, deletions, insertions
        and substiturions between reference and hypothesis string
        - Punctuation Error Rate

    Args to init:
        punctuation_marks (list[str]) - list of punctuation marks for computing metrics
        punctuation_mask (str, by default "[PUNCT]") - mask token that will be applied to
        given punctuation marks while edit distance calculation 
    
    How to use:
        1. Create object of OccurancePunctuationErrorRate class.
           Example:
                punctuation_marks = [".", ",", "!", "?"]
                oper_obj = OccurancePunctuationErrorRate(punctuation_marks)
        
        2. To compute punctuation metrics, pass reference and hypothesis string to the "compute" method
        of created object.
            Example:
                reference_str = "Hi, dear! Nice to see you. What's"
                hypothesis_str = "Hi dear! Nice to see you! What's?"
                oper_obj.compute(reference_str, hypothesis_str)

    Output (listed in order of output):
        1. Dict of absolute operations amounts for each given punctuation mark:
            Example:
            {'.': {'Correct': 0, 'Deletions': 0, 'Insertions': 0, 'Substitutions': 1},
             ',': {'Correct': 0, 'Deletions': 1, 'Insertions': 0, 'Substitutions': 0},
             '!': {'Correct': 1, 'Deletions': 0, 'Insertions': 0, 'Substitutions': 0},
             '?': {'Correct': 0, 'Deletions': 0, 'Insertions': 1, 'Substitutions': 0}}
              
        2. Dict of substitutions absolute amounts between given punctuation marks:
            Example:
            {'.': {'.': 0, ',': 0, '!': 1, '?': 0},
             ',': {'.': 0, ',': 0, '!': 0, '?': 0},
             '!': {'.': 0, ',': 0, '!': 0, '?': 0},
             '?': {'.': 0, ',': 0, '!': 0, '?': 0}}
            
        3. namedtuple "PunctuationRates" of punctuation operation rates (in range from 0 to 1):
            3.1. correct_rate - overall correct rate 
                Example: correct_rate=0.25
            3.2. deletions_rate - overall deletions rate
                Example: deletions_rate=0.25
            3.3. insertions_rate - overall insertions rate
                Example: insertions_rate=0.25
            3.4. substitutions_rate - overall substitutions_rate
                Example: substitutions_rate=0.25
            3.5. punct_er - Punctuation Error Rate
                Example: punct_er=0.75
            3.6. operation_rates - dict of operations rates for each given punctuation mark
                Example: 
                operation_rates={
                    '.': {'Correct': 0.0, 'Deletions': 0.0, 'Insertions': 0.0, 'Substitutions': 1.0},
                    ',': {'Correct': 0.0, 'Deletions': 1.0, 'Insertions': 0.0, 'Substitutions': 0.0},
                    '!': {'Correct': 1.0, 'Deletions': 0.0, 'Insertions': 0.0, 'Substitutions': 0.0},
                    '?': {'Correct': 0.0, 'Deletions': 0.0, 'Insertions': 1.0, 'Substitutions': 0.0}
                    }
  
            3.7. substitution_rates - dict of substitution rates for each given punctuation mark
                Example:
                substitution_rates={
                    '.': {'.': 0.0, ',': 0.0, '!': 1.0, '?': 0.0},
                    ',': {'.': 0.0, ',': 0.0, '!': 0.0, '?': 0.0},
                    '!': {'.': 0.0, ',': 0.0, '!': 0.0, '?': 0.0},
                    '?': {'.': 0.0, ',': 0.0, '!': 0.0, '?': 0.0}
                    }
    """

    def __init__(self, punctuation_marks: list[str], punctuation_mask: str = "[PUNCT]") -> None:

        assert len(punctuation_marks) != 0, f"List of punctuation marks is empty"

        self.punctuation_marks = punctuation_marks
        self.punctuation_mask = punctuation_mask

        self.operations = ["Correct", "Deletions", "Insertions", "Substitutions"]

    def compute_rates(self, operation_amounts: dict, substitution_amounts: dict):
        operation_rates = {pm: {operation: 0 for operation in self.operations} for pm in self.punctuation_marks}
        substitution_rates = {pm: {pm: 0 for pm in self.punctuation_marks} for pm in self.punctuation_marks}

        for pm in self.punctuation_marks:
            operations_amount_by_pm = sum(operation_amounts[pm].values())

            if operations_amount_by_pm == 0:
                continue

            operation_rates[pm] = {
                operation: (operation_amounts[pm][operation] / operations_amount_by_pm)
                for operation in self.operations
            }

            substitution_rates[pm] = {
                _pm: (substitution_amounts[pm][_pm] / operations_amount_by_pm)
                for _pm in substitution_amounts[pm].keys()
            }

        _operation_amounts = {
            operation: {pm: operation_amounts[operation] for pm, operation_amounts in operation_amounts.items()}
            for operation in self.operations
        }

        overall_amounts_by_operation = {
            operation: sum(_operation_amounts[operation].values()) for operation in _operation_amounts
        }
        overall_operations_amount = sum(overall_amounts_by_operation.values())

        punctuation_rates = namedtuple(
            'PunctuationRates',
            [
                'correct_rate',
                'deletions_rate',
                'insertions_rate',
                'substitutions_rate',
                'punct_er',
                'operation_rates',
                'substitution_rates',
            ],
        )

        if overall_operations_amount == 0:
            rates = punctuation_rates(0, 0, 0, 0, 0, operation_rates, substitution_rates)
        else:
            correct_rate = overall_amounts_by_operation["Correct"] / overall_operations_amount
            deletions_rate = overall_amounts_by_operation["Deletions"] / overall_operations_amount
            insertions_rate = overall_amounts_by_operation["Insertions"] / overall_operations_amount
            substitutions_rate = overall_amounts_by_operation["Substitutions"] / overall_operations_amount
            punct_er = deletions_rate + insertions_rate + substitutions_rate

            rates = punctuation_rates(
                correct_rate,
                deletions_rate,
                insertions_rate,
                substitutions_rate,
                punct_er,
                operation_rates,
                substitution_rates,
            )

        return rates

    def compute_operation_amounts(self, reference: str, hypothesis: str):
        operation_amounts = {pm: {operation: 0 for operation in self.operations} for pm in self.punctuation_marks}
        substitution_amounts = {pm: {pm: 0 for pm in self.punctuation_marks} for pm in self.punctuation_marks}

        def tokenize(text: str, punctuation_marks: list[str]):
            punctuation_marks = "\\" + "\\".join(self.punctuation_marks)
            tokens = re.findall(rf"[\w']+|[{punctuation_marks}]", text)
            return tokens

        def mask_punct_tokens(tokens: list[str], punctuation_marks: list[str], punctuation_mask: str):
            masked = [punctuation_mask if token in punctuation_marks else token for token in tokens]
            return masked

        r_tokens = tokenize(reference, self.punctuation_marks)
        h_tokens = tokenize(hypothesis, self.punctuation_marks)

        r_masked = mask_punct_tokens(r_tokens, self.punctuation_marks, self.punctuation_mask)
        h_masked = mask_punct_tokens(h_tokens, self.punctuation_marks, self.punctuation_mask)

        r_punct_amount = r_masked.count(self.punctuation_mask)
        h_punct_amount = h_masked.count(self.punctuation_mask)

        if r_punct_amount + h_punct_amount == 0:
            return operation_amounts, substitution_amounts

        r_len = len(r_masked)
        h_len = len(h_masked)

        costs = [[0 for inner in range(h_len + 1)] for outer in range(r_len + 1)]
        backtrace = [[0 for inner in range(h_len + 1)] for outer in range(r_len + 1)]

        COR = 'C'
        DEL, DEL_PENALTY = 'D', 1
        INS, INS_PENALTY = 'I', 1
        SUB, SUB_PENALTY = 'S', 1

        for i in range(1, r_len + 1):
            costs[i][0] = DEL_PENALTY * i
            backtrace[i][0] = DEL

        for j in range(1, h_len + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = INS

        for j in range(1, h_len + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = INS

        for i in range(1, r_len + 1):
            for j in range(1, h_len + 1):
                if r_masked[i - 1] == h_masked[j - 1]:
                    costs[i][j] = costs[i - 1][j - 1]
                    backtrace[i][j] = COR
                else:
                    substitution_cost = costs[i - 1][j - 1] + SUB_PENALTY
                    insertion_cost = costs[i][j - 1] + INS_PENALTY
                    deletion_cost = costs[i - 1][j] + DEL_PENALTY

                    costs[i][j] = min(substitution_cost, insertion_cost, deletion_cost)
                    if costs[i][j] == substitution_cost:
                        backtrace[i][j] = SUB
                    elif costs[i][j] == insertion_cost:
                        backtrace[i][j] = INS
                    else:
                        backtrace[i][j] = DEL

        i = r_len
        j = h_len

        while i > 0 or j > 0:
            if backtrace[i][j] == COR:
                if r_masked[i - 1] == self.punctuation_mask or h_masked[j - 1] == self.punctuation_mask:
                    r_token = r_tokens[i - 1]
                    h_token = h_tokens[j - 1]

                    if r_token == h_token:
                        operation_amounts[r_token]['Correct'] += 1
                    else:
                        operation_amounts[r_token]['Substitutions'] += 1
                        substitution_amounts[r_token][h_token] += 1
                i -= 1
                j -= 1

            elif backtrace[i][j] == SUB:
                i -= 1
                j -= 1

            elif backtrace[i][j] == INS:
                j -= 1

            elif backtrace[i][j] == DEL:
                i -= 1

        for pm in self.punctuation_marks:
            num_of_correct = operation_amounts[pm]['Correct']

            num_substitutions_of_pm = operation_amounts[pm]['Substitutions']
            num_substitutions_to_pm = sum([substitution_amounts[_pm][pm] for _pm in self.punctuation_marks])

            num_of_deletions = r_tokens.count(pm) - (num_of_correct + num_substitutions_of_pm)
            operation_amounts[pm]['Deletions'] = num_of_deletions

            num_of_insertions = h_tokens.count(pm) - (num_of_correct + num_substitutions_to_pm)
            operation_amounts[pm]['Insertions'] = num_of_insertions

        return operation_amounts, substitution_amounts

    def compute(self, reference: str, hypothesis: str):
        operation_amounts, substitution_amounts = self.compute_operation_amounts(reference, hypothesis)
        punctuation_rates = self.compute_rates(operation_amounts, substitution_amounts)
        return operation_amounts, substitution_amounts, punctuation_rates


class DatasetPunctuationErrorRate:
    """
    Class for computation the total puncutation-related absolute amounts of operations and their rates 
    in pairs of reference and hypothesis strins:
        - Absolute amounts of correct predictions, deletions, insertions
        and substitutions for each given punctuation mark
        - Rates of correct predictions, deletions, insertions
        and substitutions for each given punctuation mark 
        - Total rates of correct predictions, deletions, insertions
        and substiturions in pairs of reference and hypothesis strings 
        - Punctuation Error Rate
        
    Args to init:
        references (list[str]) - list of references
        hypotheses (list[str]) - list of hypotheses
        punctuation_marks (list[str]) - list of punctuation marks for computing metrics
        punctuation_mask (str, by default "[PUNCT]") - mask token that will be applied to
        given punctuation marks while edit distance calculation
        
    How to use:
        1. Create object of DatasetPunctuationErrorRate class.
           Example:
                references = ["Hi, dear! Nice to see you. What's"]
                hypotheses = ["Hi dear! Nice to see you! What's?"]                
                punctuation_marks = [".", ",", "!", "?"]
                
                dper_obj = DatasetPunctuationErrorRate(references, hypotheses, punctuation_marks)
                
        2. To compute punctuation metrics, call the class method "compute()".
            Example:
                dper_obj.compute() 
                
    Result:
    The following atributes of class object will be updated with calculated metrics values.
    The values are available with calling the atributes:
        
        dper_obj.operation_rates - dict, rates of correctness and errors for each punctuation mark 
        from `preset dper_obj.punctuation_marks` list.
        
        dper_obj.substitution_rates - dict, substitution rates between puncutation marks from
        `preset dper_obj.punctuation_marks` list.
        
        dper_obj.correct_rate - float, total rate of correctness between provided pairs of 
        references and hypotheses.
        
        dper_obj.deletions_rate - float, total rate of deletions between provided pairs of 
        references and hypotheses.
        
        dper_obj.insertions_rate - float, total rate of insertions between provided pairs of 
        references and hypotheses.
        
        dper_obj.substitutions_rate - float, total rate of substitutions between provided pairs of 
        references and hypotheses.
        
        dper_obj.punct_er - float, total Punctuation Error Rate between provided pairs of 
        references and hypotheses.
    """

    def __init__(
        self,
        references: list[str],
        hypotheses: list[str],
        punctuation_marks: list[str],
        punctuation_mask: str = "[PUNCT]",
    ) -> None:

        self.references = references
        self.hypotheses = hypotheses
        self.punctuation_marks = punctuation_marks
        self.punctuation_mask = punctuation_mask

        self.oper_obj = OccurancePunctuationErrorRate(
            punctuation_marks=self.punctuation_marks, punctuation_mask=self.punctuation_mask
        )

        self.operation_amounts = []
        self.substitution_amounts = []
        self.rates = []

        self.operation_rates = None
        self.substitution_rates = None
        self.correct_rate = None
        self.deletions_rate = None
        self.insertions_rate = None
        self.substitutions_rate = None
        self.punct_er = None

    def compute(self):
        def sum_amounts(amounts_dicts: list[dict]):
            amounts = {key: {_key: 0 for _key in amounts_dicts[0][key]} for key in amounts_dicts[0].keys()}

            for amounts_dict in amounts_dicts:
                for outer_key, inner_dict in amounts_dict.items():
                    for inner_key, value in inner_dict.items():
                        amounts[outer_key][inner_key] += value
            return amounts

        logging.info("Computing Punctuation Error Rate")

        for reference, hypothesis in tqdm(zip(self.references, self.hypotheses), total=len(self.references)):
            operation_amounts, substitution_amounts, punctuation_rates = self.oper_obj.compute(reference, hypothesis)
            self.operation_amounts.append(operation_amounts)
            self.substitution_amounts.append(substitution_amounts)
            self.rates.append(punctuation_rates)

        overall_operation_amounts = sum_amounts(self.operation_amounts)
        overall_substitution_amounts = sum_amounts(self.substitution_amounts)
        overall_rates = self.oper_obj.compute_rates(
            operation_amounts=overall_operation_amounts, substitution_amounts=overall_substitution_amounts
        )

        self.operation_rates = overall_rates.operation_rates
        self.substitution_rates = overall_rates.substitution_rates
        self.correct_rate = overall_rates.correct_rate
        self.deletions_rate = overall_rates.deletions_rate
        self.insertions_rate = overall_rates.insertions_rate
        self.substitutions_rate = overall_rates.substitutions_rate
        self.punct_er = overall_rates.punct_er

    def reset(self):
        self.operation_amounts = []
        self.substitution_amounts = []
        self.rates = []

        self.operation_rates = None
        self.substitution_rates = None
        self.correct_rate = None
        self.deletions_rate = None
        self.insertions_rate = None
        self.substitutions_rate = None
        self.punct_er = None

    def print(self):
        logging.info(f'Dataset PER ' + str(round(100 * self.punct_er, 2)) + '%')

        if HAVE_TABLUATE_AND_PANDAS:
            rates_by_pm_df = pd.DataFrame(self.operation_rates) * 100
            substitution_rates_by_pm_df = pd.DataFrame(self.substitution_rates) * 100

            logging.info(
                "Rates of punctuation correctness and errors (%):\n"
                + tabulate(rates_by_pm_df, headers='keys', tablefmt='psql')
            )
            logging.info(
                "Substitution rates between punctuation marks (%):\n"
                + tabulate(substitution_rates_by_pm_df, headers='keys', tablefmt='psql')
            )
        else:
            logging.warning("Some of the modules (pandas or tabulate) can't be imported")
            logging.info(f"Rates of punctuation correctness and errors (in range [0, 1]):\n{self.operation_rates}\n")
            logging.info(
                f"Substitution rates between punctuation marks (in range [0, 1]):\n{self.substitution_rates}\n"
            )
