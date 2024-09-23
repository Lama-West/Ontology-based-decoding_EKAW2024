# Source
from typing import List

import numpy as np
from transformers import pipeline

class NLIAnalyzer:
    """
    Comparative NLI Analyzer.
    """

    def __init__(self, nli_model_path: str):
        self.classifier = pipeline("zero-shot-classification",
            model=nli_model_path, 
            local_files_only=True,
            device_map='cuda'
        )

    def evaluate(self, premise: str, hypotheses: List[str], multi_label=True):
        return self.classifier(premise, hypotheses, multi_label=multi_label, batch_size=8)

    def compare_statements(self, document: str, statements: List[List[str]]):
        """
        Compares the NLI score of multiple groups on a document. Will take the best statement
        at each step and compute the number of times a group has won

        Args :
            - document    :   Document to ground the statements to
            - statements  :   List of list of statements. It should have the following size [nb_groups, nb_statements]
        """       

        statements_per_group = list(map(list, zip(*statements)))
        
        results = []
        all_scores = []
        for group_statement in statements_per_group:

            if 'N/A' in group_statement:
                continue

            formatted_statements = []
            for s in group_statement:
                if 'N/A' in s:
                    continue
                formatted_statements.append(s.replace('. ', '').strip().replace(',', ''))

            if len(document) > 1536:
                # print('Skipping document')
                continue

            output = self.classifier(document, formatted_statements, multi_label=True)
            if np.all(np.isclose(output['scores'], output['scores'][0])):
                best_statement = -1
            else:
                best_statement = np.argmax(output['scores'])
    
            all_scores.append(output['scores'])
            results.append(best_statement)

        return all_scores, self.compute_nli_score(results, all_scores)

    def compute_nli_score(self, results, all_scores):
        """
        Computes the nli score by taking the average win score of all results.
        If a group A has the highest score for a statement, the difference between
        its score and the average of all groups will be added to its score. At the end,
        all scores are averaged by the number of statements (length of results)
        """
        if len(all_scores) == 0 or len(results) == 0:
            return 0

        scores = [0 for _ in range(len(all_scores[0]))]

        count = 0
        for i, result in enumerate(results):

            # Don't consider if it's a tie
            if result == -1:
                continue
            count += 1
            
            # The score of the winner is increase by its value minus the average
            scores[result] += len(all_scores[i]) * (all_scores[i][result] - np.mean(all_scores[i]))

        if count == 0:
            return scores

        # Take the average
        for i in range(len(scores)):
            scores[i] /= count

        return scores

    def evaluate_document(self, document: str, statements: List[List[str]]):
        """
        Evaluates the NLI score of multiple statements on a document.

        Args 
            - statements  :   List of list of statements. It should have the following size [batch_size, nb_statements]
        """

        results = []
        for batch in statements:
            
            formatted_batch = []
            for b in batch:
                if 'N/A' in b:
                    continue
                formatted_batch.append(b.replace('. ', '').strip().replace(',', ''))

            result = self.classifier(document, formatted_batch, multi_label=True)
            scores = result['scores']
            results.append(np.mean(scores))
        return results
