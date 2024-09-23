

import nltk.translate.bleu_score
from transformers import LogitsProcessor, BeamSearchScorer
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

from typing import Dict, List, Optional, Tuple, Union, ClassVar
from dataclasses import dataclass, field
import time
import re

import nltk
import torch

from src.ontology.umls import Annotator
from src.ontology.snomed import SNOMED

@dataclass
class DiverseBeamSearchConfig:

    max_length: int = 4096
    use_scorer: bool = False
    normal_beam_search: bool = False
    nb_beams: int = 4
    nb_beam_groups: int = 2
    nb_beam_hyps_to_keep: int = 1
    window_size: int = 5
    temperature: float = 1.0
    diversity_penalty: float = 1e7
    
    # Boost factor used multiplied by the frequency when a concept is present. This 
    # parameter puts more emphasis on generating classes that are children of the 
    # base concept
    # boost_factor: float = 1000.0

    # Base beam boost if a concept is found. This value is added to the frequencies 
    # if present in the frequencies dictionary. This is used to put more emphasis on
    # generating medical concepts
    hierarchy_score_boost = 1.0

    # If the property value (see `get_value()` of the properties class) has a certain
    # ressemblance to the context measure by the ROUGE metric, the bleu score is added
    # to the property score. This cutoff is used to control at which rouge score we
    # consider the rouge score to be high enough to have an effect on the beam score.
    # Not considered anymore
    property_cutoff: float = 0.0

    # Weights associated with the original beam scores and the boost scores.
    score_weights: Tuple[float, float] = (0.0, 1.0)

    # Corresponds to H_bf, P_bf and S_bf
    score_boost_factors = [3.0, 1.0, 10.0]

    # The exclude_ids list contains classes that are not entirely linked to concepts
    # but more like values or environments. This is usually what is present in the 
    # answer. They should be excluded in the prompting process, but are important in
    # the decoding process
    exclude_ids: ClassVar[set[str]] = set(['362981000', '419891008', '106237007'])
    

# group_size = nb_beams / nb_beam_groups
# input_size = batch_size * group_size

# src: https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html
def set_scores_to_inf_for_banned_tokens(scores, banned_tokens):
    """
    Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
    list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

    Args:
        scores: logits distribution of shape (batch size, vocabulary size)
        banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return scores

    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))

    banned_mask = (
        torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    )
    scores = scores.masked_fill(banned_mask, -float("inf"))
    return scores

class MedicalBeamScorerConfig:
    # TODO : Generation config has a lot of the same attributes as the BeamSearchScorer class. Simplify it
    def __init__(
        self, 
        tokenizer, 
        annotator: Annotator, 
        snomed: SNOMED, 
        base_class_ids: List[str], 
        generation_config: DiverseBeamSearchConfig
    ) -> None:
        self.tokenizer = tokenizer
        self.annotator = annotator
        self.snomed = snomed
        self.base_class_ids = base_class_ids
        self.generation_config = generation_config
        # TODO : Remove temperature since useless
        assert self.generation_config.temperature > 0, "Temperature cannot be null"


class MedicalBeamScorer(BeamSearchScorer):
    def __init__(
        self,
        config: MedicalBeamScorerConfig,
        clinical_note: str,
        batch_size: int, 
        device: torch.device, 
        length_penalty: Optional[float] = 1.0, 
        do_early_stopping: Optional[Union[bool, str]] = False, 
    ):
        super().__init__(
            batch_size, 
            config.generation_config.nb_beams, 
            device, 
            length_penalty, 
            do_early_stopping, 
            config.generation_config.nb_beam_hyps_to_keep, 
            config.generation_config.nb_beam_groups, 
            config.generation_config.max_length
        )
        self.clinical_note = clinical_note
        self.batch_size = batch_size
        self.config = config
        self.nb_tokens_generated = self.config.generation_config.window_size // 2

        self.time_to_next_process = -1
        self.nb_times_process_called = 0
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)


    def get_base_class_id_from_index(self, index):
        """
        Retrieves the associated base class of the query given an 
        in index from a batched, beamed input_id tensor
        """
        batch_index = index % self.batch_size
        return self.config.base_class_ids[batch_index]

    def id_to_label(self, id: str):
        if id in self.config.snomed.id_to_classes:
            return self.config.snomed.get_class_from_id(id).label
        return 'N/A'

    def get_hierarchy_beam_boost(self, base_class_id: str, detected_class_id: str):
        """
        Args
            - base_class_id       : Id of base class related to the query
            - detected_class_id   : Id of the concept detected in the answer
        """

        if detected_class_id not in self.config.snomed.id_to_classes:
            return 0

        if detected_class_id == base_class_id:
            return 0 #self.config.generation_config.hierarchy_score_boost

        parents = self.config.snomed.get_ancestors_of_id(detected_class_id, return_list=True)
        parents_in_exclusion = set(parents).intersection(DiverseBeamSearchConfig.exclude_ids)
        if len(parents_in_exclusion) > 0:
            # The exclude_ids list contains classes that are not entirely linked to concepts
            # but mot like values or environments. This is usually what is present in the 
            # answer
            return 0 # self.config.generation_config.hierarchy_score_boost
        
        if base_class_id not in parents:
            # We don't want concepts from other branches
            return 0 # -self.config.generation_config.hierarchy_score_boost
        
        return self.config.generation_config.hierarchy_score_boost

    def get_similarity_beam_boost(self, context: str):
        """
        Computes the similarity score. It compares the current context beam to the clinical notes to encourage
        beams to ressemble the formulation of the clinical notes

        Args
            - context   : Context of the current beam
        """
        # We use every sentence of the clinical notes as a reference
        references = self.clinical_note.split('. ')

        # Compute ROUGE-2
        beam_boost = max(map(lambda x: self.rouge_scorer.score(context, x)['rouge2'].precision, references))
        return beam_boost

    def get_properties_beam_boost(self, base_class_id: str, detected_class_id: str, context: str):
        """
        Computes property score.

        Args
            - base_class_id       : Id of base class related to the query
            - detected_class_id   : Id of the concept detected in the answer
            - context             : Context where `detected_class_id` was detected
        """
        
        properties = self.config.snomed.get_properties_of_id(base_class_id)
        if len(properties) == 0:
            return 0
        
        property_score = 0

        # Direct property link in ontology
        for property in properties:
            for k, v in property.ids_to_ids.items():
                if k == detected_class_id or v == detected_class_id:
                    # Detected concept id is directly linked to the base class id
                    property_score += 1 / (len(property.ids_to_ids))

                detected_class_ancestors = self.config.snomed.get_ancestors_of_id(detected_class_id, return_set=True)
                if v in detected_class_ancestors:
                    property_score += 1 / (len(property.ids_to_ids))
                
        # Indirect property link : Add the rouge score between all property values and the context
        current_property_knowledge = ' '.join(map(lambda x: x.get_value(), properties))        
        rouge_score = self.rouge_scorer.score(context, current_property_knowledge)['rouge2'].precision

        if rouge_score > self.config.generation_config.property_cutoff:
            property_score += rouge_score

        # We divide by two since the direct property link is between 0 and 1
        # and the indirect property link is between 0 and 1. Thus the max value
        # of the property score is 2
        return property_score / 2 

    def get_beam_boost(self, input_ids, group_index):
        """
        Computes the beam score based on all scores (hierarchy, property, similarity)
        """

        context_size = int(2 * self.config.generation_config.window_size)
        decoded_next_tokens = self.config.tokenizer.batch_decode(input_ids[:, -context_size:])
        batched_annotations = self.config.annotator.batch_annotate(decoded_next_tokens, return_ids_only=True)
        scores = []
        for i, annotations in enumerate(batched_annotations):
            base_class_id = self.get_base_class_id_from_index(i)
            decoded_context = decoded_next_tokens[i]
            if 'N/A' in decoded_context:
                scores.append(0.0)
                continue
            avg_hierarchy_score = 0
            avg_property_score = 0
            for snomed_id in annotations:
                avg_hierarchy_score += self.get_hierarchy_beam_boost(base_class_id, snomed_id)
                avg_property_score += self.get_properties_beam_boost(base_class_id, snomed_id, decoded_context)

            avg_hierarchy_score /= max(1, len(annotations))
            avg_property_score /= max(1, len(annotations))
            groundedness_score = self.get_similarity_beam_boost(decoded_context)
            freq_adapted_score = \
                  self.config.generation_config.score_boost_factors[0] * avg_hierarchy_score \
                + self.config.generation_config.score_boost_factors[1] * avg_property_score \
                + self.config.generation_config.score_boost_factors[2] * groundedness_score
            scores.append(freq_adapted_score)

        return scores
    
    def compute_new_beam_scores(self, current_scores, boost_factors):
        """
        Computes the new beam score
        """
        boost_factors = torch.tensor(
            boost_factors, 
            device=current_scores.device, 
            dtype=torch.float32
        )

        boost_factors = torch.nn.functional.log_softmax(boost_factors, dim=-1)

        weights = self.config.generation_config.score_weights
        final_results = (weights[0] * current_scores + weights[1] * boost_factors) / (weights[0] + weights[1])
        del boost_factors
        return final_results

    def process(
        self, 
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Process beam to update beam scores
        """
        if self.time_to_next_process < 0:
            self.time_to_next_process = time.time()
        else:
            self.time_to_next_process = time.time()
        
        results = super().process(input_ids, next_scores, next_tokens, next_indices, pad_token_id, eos_token_id, beam_indices, group_index, decoder_prompt_len)

        if self.config.generation_config.normal_beam_search:
            return results

        self.nb_times_process_called += 1
        modifying_scores: bool = self.nb_tokens_generated >= self.config.generation_config.window_size
        if modifying_scores:
            # Custom process
            old_beam_scores = results['next_beam_scores']
            boost_factors = self.get_beam_boost(input_ids, group_index=group_index)
            results['next_beam_scores'] = self.compute_new_beam_scores(old_beam_scores, boost_factors)

            # The beam search scorer will be called for each group.
            # Thus, every `window_size` tokens, we need to process the next
            # `nb_group` calls to this function
            if group_index >= self.config.generation_config.nb_beam_groups - 1:
                self.nb_tokens_generated = 1 # While we modified the scores, a token was processed
        else:
            if group_index >= self.config.generation_config.nb_beam_groups - 1:
                # We increase the number of tokens generated only if we have processed
                # every group since this function is called for every group for every token
                self.nb_tokens_generated += 1

        return results
    
    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Tuple[torch.LongTensor]:
        """
        Finalize beam algorithm
        """
        return super().finalize(input_ids, final_beam_scores, final_beam_tokens, final_beam_indices, max_length, pad_token_id, eos_token_id, beam_indices, decoder_prompt_len)
