import random
from typing import Dict, List
from collections import Counter
import ast
import os

from tqdm import tqdm
from colorist import Color

import torch
from transformers import (
    LogitsProcessorList, 
    StoppingCriteriaList,
    MaxLengthCriteria, 
)
from accelerate import Accelerator
import pandas as pd

from src.hf_utils import clear_gpu_cache
from src.domain_class_frequency import DomainClassFrequency
from src.medical_beam_scorer import MedicalBeamScorer, MedicalBeamScorerConfig, DiverseBeamSearchConfig
from src.preprocessor import preprocess_clinical_notes
from src.chat_template import ChatTemplate
from src.ontology.umls import Annotator, UMLSAnnotatorResult
from src.ontology.snomed import SNOMED
from src.prompts import BASE_PROMPT_TEMPLATE

class OntologyConstrainedModel:
    """
    Model constrained by ontology during decoding process. Can't be used for inference
    normally without constrained decoding process
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        snomed: SNOMED, 
        annotator: Annotator, 
        accelerator: Accelerator = None
    ) -> None:
        """
        Args
            - model         : Model to use for inference (Must be AutoModelFromCausalLM)
            - tokenizer     : Tokenizer linked to the model
            - snomed        : SNOMED object used to retrieve links between concepts
            - annotator     : Annotator that can tag a text to retrieve ontological concepts
            - accelerator   : Accelerator object to parallelize operation
        """

        self.model = model
        self.tokenizer = tokenizer

        self.chat_template = ChatTemplate(tokenizer)
        self.snomed = snomed
        self.annotator = annotator
        self.accelerator = accelerator
    
    def get_device(self):
        """
        Returns the device that should be used for inference
        """
        return self.model.device if self.accelerator is None else self.accelerator.device

    def prepare_model_inputs(self, questions: List[str]):
        """
        Prepares a list of questions to be sent to the model. It applies these transformations :
        - Applies the chat template of the model
        - Tokenizes the questions

        Args :
            - questions : List of questions to be sent to the model

        Returns
        Questions in chat template tokenized
        """
        prompts = list(map(self.chat_template.single_user_entry, questions))
        model_input = self.tokenizer(
            prompts, 
            padding=True, 
            return_tensors="pt",
            truncation=False, 
            pad_to_multiple_of=8,
            add_special_tokens=False
        )
        return model_input

    def prepare_input_for_beam_search(self, tensor: torch.Tensor, expand_size: int, device) -> torch.Tensor:
        """
        Interleaves the data to prepare the input for beam search

        Args :
            - tensor        : Input tensor to be sent to the model
            - expand_size   : Expand size of tensor (should be the number of beams)
        """
        return torch.repeat_interleave(tensor, expand_size, dim=0).to(device)
    
    def format_generated_answer(self, prompts_input_ids: torch.Tensor, generated_answer) -> List[str]:
        """
        Formats the generated answer by removing the initial prompt sent to the model and decoding the answer

        Args :
            - prompt_input_ids  : Initial prompt input_ids sent to the model to get the generation
            - generated_answer  : Generated tokens of the model
        """
        new_tokens = generated_answer[:, prompts_input_ids.shape[-1]:]
        results = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return results


    def ask_without_scorer(self, questions: List[str], generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()) -> List[str]:
        """
        Sents a list of questions to the model without using grouped beam search

        Args :
            - questions         : Questions to send to the model
            - generation_config : Generation config used during inference

        Returns
        Generated answers (decoded)
        """
        model_input = self.prepare_model_inputs(questions).to(self.get_device())
        self.model.eval()
        with torch.no_grad():
            generated = self.generate(model_input, scorer=None, generation_config=generation_config)
            final_answers = self.format_generated_answer(model_input['input_ids'], generated)
            del model_input
            return final_answers
    
    def ask_with_scorer(self, questions: List[str], concept_ids: List[str] = [], generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()):
        """
        Sents a list of questions to the model using grouped beam search

        Args :
            - questions         : Questions to send to the model
            - concept_ids       : Ids of concepts in ontology that will be used to guide the beam search algorithm
            - generation_config : Generation config used during inference

        Returns
        Generated answers (decoded)
        """
        # TODO : Implement batched note answering
        # Right now, questions can be batched, but only according to a single clinical note
        batch_size = len(questions)
        medical_beam_scorer = MedicalBeamScorer(
            clinical_note=questions[0], # The clinical note is the same across all questions
            config=MedicalBeamScorerConfig(
                tokenizer=self.tokenizer,
                annotator=self.annotator,
                snomed=self.snomed,
                base_class_ids=concept_ids,
                generation_config=generation_config
            ),
            batch_size=batch_size,
            device=self.get_device(),
        )

        prompts_tokenized = self.prepare_model_inputs(questions)        

        prompts_tokenized['input_ids'] = self.prepare_input_for_beam_search(
            prompts_tokenized['input_ids'], 
            device=self.get_device(),
            expand_size=generation_config.nb_beams
        )

        prompts_tokenized['attention_mask'] = self.prepare_input_for_beam_search(
            prompts_tokenized['attention_mask'], 
            device=self.get_device(),
            expand_size=generation_config.nb_beams
        )
        
        self.model.eval()
        with torch.no_grad():
            generated = self.generate(
                prompts_tokenized, 
                scorer=medical_beam_scorer, 
                generation_config=generation_config
            )
            final_answers = self.format_generated_answer(prompts_tokenized['input_ids'], generated)

            del prompts_tokenized
            
            return final_answers

    def generate(self, model_input: Dict[str, torch.Tensor], scorer = None, generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()):
        """
        Sends an input (input_ids + attention_mask) to the model for inference using a generation config

        Args
            - model_input       : Dictionary containing the input ids and attention masks
            - scorer            : Scorer to use during generation
            - generation_config : Generation config to use during inference
        """
        if generation_config.use_scorer:
            config_dict = {
                'num_beams': generation_config.nb_beams,
                'num_beam_groups': generation_config.nb_beam_groups,
                # 'temperature': generation_config.temperature,
                'do_sample': False,
                'diversity_penalty': generation_config.diversity_penalty,
                'num_return_sequences': 1,
                'max_length': generation_config.max_length,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            hf_gen_config, _ = self.model._prepare_generation_config(None, **config_dict)
            
            logits_processor = LogitsProcessorList([])
            logits_warper = self.model._get_logits_processor(
                generation_config=hf_gen_config,
                input_ids_seq_length=model_input['input_ids'].shape[-1],
                encoder_input_ids=model_input['input_ids'],
                prefix_allowed_tokens_fn=None,
                logits_processor=[],
            )

            return self.model._group_beam_search(
                **model_input,
                beam_scorer=scorer,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                stopping_criteria=StoppingCriteriaList([
                    MaxLengthCriteria(max_length=generation_config.max_length),
                ]),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return self.model.generate(
            **model_input, 
            max_new_tokens=128,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def ask(self, questions: List[str], concept_id: List[str] = [], generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()):
        """
        Asks the model a list of question using a list of concepts. If `generation_config.use_scorer` is False, then `concept_id` is not used

        Args :
            - questions         : Questions to send to the model
            - concept_id       : Ids of concepts in ontology that will be used to guide the beam search algorithm
            - generation_config : Generation config used during inference
        """
        if not generation_config.use_scorer:
            return self.ask_without_scorer(questions, generation_config=generation_config)
        return self.ask_with_scorer(questions, concept_id, generation_config=generation_config)

class SimpleOntologyConstrainedModel:
    """
    Same as `OntologyConstrainedModel`, but assumes that data is already tokenized. Useful when
    working with multiple GPUs since tokenization can be done beforehand. 
    TODO : Should inherit OntologyConstrainedModel, although not trivial since signature changes
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        snomed: SNOMED, 
        annotator: Annotator, 
        accelerator: Accelerator = None
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer

        self.chat_template = ChatTemplate(tokenizer)
        self.snomed = snomed
        self.annotator = annotator
        self.accelerator = accelerator
    
    def get_device(self):
        """
        Returns the device that should be used for inference
        """

    def prepare_input_for_beam_search(self, tensor: torch.Tensor, expand_size: int, device) -> torch.Tensor:
        """
        Interleaves the data to prepare the input for beam search

        Args :
            - tensor        : Input tensor to be sent to the model
            - expand_size   : Expand size of tensor (should be the number of beams)
        """
        return torch.repeat_interleave(tensor, expand_size, dim=0).to(device)
    
    def format_generated_answer(self, prompts_input_ids: torch.Tensor, generated_answer) -> List[str]:
        """
        Formats the generated answer by removing the initial prompt sent to the model and decoding the answer

        Args :
            - prompt_input_ids  : Initial prompt input_ids sent to the model to get the generation
            - generated_answer  : Generated tokens of the model
        """
        new_tokens = generated_answer[:, prompts_input_ids.shape[-1]:]
        results = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return results

    def ask_without_scorer(self, input_ids, attention_mask, questions: List[str], generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()) -> List[str]:
        """
        Sents a list of questions to the model without using grouped beam search

        Args :
            - questions         : Questions to send to the model
            - generation_config : Generation config used during inference

        Returns
        Generated answers (decoded)
        """
        self.model.eval()
        with torch.no_grad():
            generated = self.generate(
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }, 
                scorer=None, 
                generation_config=generation_config
            )
            final_answers = self.format_generated_answer(input_ids, generated)
            return final_answers
    
    def ask_with_scorer(self, input_ids, attention_mask, questions: List[str], concept_ids: List[str] = [], generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()):
        """
        Sents a list of questions to the model using grouped beam search

        Args :
            - questions         : Questions to send to the model
            - concept_ids       : Ids of concepts in ontology that will be used to guide the beam search algorithm
            - generation_config : Generation config used during inference

        Returns
        Generated answers (decoded)
        """
        batch_size = len(questions)

        medical_beam_scorer = MedicalBeamScorer(
            clinical_note=questions[0],
            config=MedicalBeamScorerConfig(
                tokenizer=self.tokenizer,
                annotator=self.annotator,
                snomed=self.snomed,
                base_class_ids=concept_ids,
                generation_config=generation_config
            ),
            batch_size=batch_size,
            device=self.get_device(),
        )

        input_ids = self.prepare_input_for_beam_search(
            input_ids, 
            device=self.get_device(),
            expand_size=generation_config.nb_beams
        )

        attention_mask = self.prepare_input_for_beam_search(
            attention_mask, 
            device=self.get_device(),
            expand_size=generation_config.nb_beams
        )
        
        self.model.eval()
        with torch.no_grad():
            generated = self.generate(
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }, 
                scorer=medical_beam_scorer, 
                generation_config=generation_config
            )
            final_answers = self.format_generated_answer(input_ids, generated)
            return final_answers

    def generate(self, model_input: Dict[str, torch.Tensor], scorer = None, generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()):
        """
        Sends an input (input_ids + attention_mask) to the model for inference using a generation config

        Args
            - model_input       : Dictionary containing the input ids and attention masks
            - scorer            : Scorer to use during generation
            - generation_config : Generation config to use during inference
        """
        if generation_config.use_scorer:
            config_dict = {
                'num_beams': generation_config.nb_beams,
                'num_beam_groups': generation_config.nb_beam_groups,
                # 'temperature': generation_config.temperature,
                'do_sample': False,
                'diversity_penalty': generation_config.diversity_penalty,
                'num_return_sequences': 1,
                'max_length': generation_config.max_length,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            hf_gen_config, _ = self.model._prepare_generation_config(None, **config_dict)
            
            logits_processor = LogitsProcessorList([])
            logits_warper = self.model._get_logits_processor(
                generation_config=hf_gen_config,
                input_ids_seq_length=model_input['input_ids'].shape[-1],
                encoder_input_ids=model_input['input_ids'],
                prefix_allowed_tokens_fn=None,
                logits_processor=[],
            )

            return self.model._group_beam_search(
                **model_input,
                beam_scorer=scorer,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                stopping_criteria=StoppingCriteriaList([
                    MaxLengthCriteria(max_length=generation_config.max_length),
                ]),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return self.model.generate(
            **model_input, 
            max_new_tokens=128,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def ask(
        self, 
        input_ids, 
        attention_mask, 
        questions: List[str], 
        concept_id: List[str] = [], 
        generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()
    ):
        """
        Asks the model a list of question using a list of concepts. If `generation_config.use_scorer` is False, then `concept_id` is not used

        Args :
            - questions         : Questions to send to the model
            - concept_id       : Ids of concepts in ontology that will be used to guide the beam search algorithm
            - generation_config : Generation config used during inference
        """
        if not generation_config.use_scorer:
            return self.ask_without_scorer(input_ids, attention_mask, questions, generation_config=generation_config)
        return self.ask_with_scorer(input_ids, attention_mask, questions, concept_id, generation_config=generation_config)


class PandasMockConstrainedModel(OntologyConstrainedModel):
    """
    Mock of OntologyConstrainedModel which simply stores the prompts to be ran in the future. This prevents the full prompting technique
    to be ran at the same time as the model's inference. We store the prompts and will ask the prompts later on
    """


    def __init__(self, snomed: SNOMED, annotator: Annotator, accelerator: Accelerator = None) -> None:
        super().__init__(None, None, snomed, annotator, accelerator)
        self.data = pd.DataFrame(columns = ['hadm_id', 'note_id', 'prompt', 'concept_id'])
    
    def prepare(self, hadm_id, note_id):
        self.hadm_id = hadm_id
        self.note_id = note_id

    def ask(self, questions: List[str], concept_id: List[str] = [], generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()):
        # Only single batching is supported for now
        assert self.hadm_id is not None and self.note_id is not None, "Prepared was not called before asking"
        prompt = questions[0]
        cid = concept_id[0]
        self.data = pd.concat([self.data, pd.DataFrame([{
            'hadm_id': self.hadm_id,
            'note_id': self.note_id,
            'prompt': prompt,
            'concept_id': cid
        }])])
        return ['Mocked answer'] * len(questions)


class OntologyPromptTemplate:

    def __init__(self, question_template: str = None):
        if question_template is None:
            self.question_template = BASE_PROMPT_TEMPLATE
        else:
            self.question_template = question_template

class OntologyBasedPrompter:
    """
    Responsable of the extraction step
    """
    
    def __init__(
        self, 
        constrained_model: OntologyConstrainedModel, 
        snomed_ct: SNOMED, 
        annotator: Annotator, 
        use_properties: bool = True,
        template: OntologyPromptTemplate = OntologyPromptTemplate()
    ):
        
        """
        Args :
        - constrained_model : Constrained model used for inference
        - snomed_ct         : SNOMED ontology
        - annotator         : Annotator that can tag concepts in a text to the ontology
        - use_properties    : Whether to use properties to construct prompt
        - template          : Base template for asking. Must contain
        """
        
        self.constrained_model = constrained_model
        self.snomed_ct = snomed_ct
        self.annotator = annotator
        self.use_properties = use_properties
        self.template = template

        self.attributes = list()
        self.attributes_by_id = list()
        # self.path = []
        self.full_exclude_ids = set([self.snomed_ct.base_class.id, '362981000', '123037004', '276339004', '106237007'])
        self.exclude_ids = set(['362981000', '444677008', '419891008', '276339004', '106237007'])
        self.current_note_id = 0

    def normalize_frequency_dict(self, frequency_dict: Dict[str, int]):
        """
        Takes a dictionary where the keys are a string and the values are the occurrence
        of that string (in a text for example) and normalizes the occurrence using the
        total occurrences
        """
        frequencies = {}

        total = 0
        for key, count in frequency_dict.items():
            total += count
            frequencies[key] = count

        for key, value in frequencies.items():
            frequencies[key] = value / total

        return frequencies

    def get_all_ancestors_and_ids(self, ids):
        """
        Computes the ancestor of all ids and returns the ids and their first parent
        (excluding ids from `self.exclude_ids`)
        Args :
            - ids : List of SNOMED CT ids
        """
        
        all_ids = []
        for snomed_id in ids:
            ancs = self.snomed_ct.get_ancestors_of_id(snomed_id, return_list=True)
            to_add = len(ancs) > 0
            for anc in ancs:
                if anc in self.exclude_ids:
                    to_add = False
                    break
            if to_add:
                if len(ancs) > 0:
                    all_ids.append(ancs[0])
                all_ids.append(snomed_id)
        return all_ids
    
    def id_to_label(self, id: str):
        if id in self.snomed_ct.id_to_classes:
            return self.snomed_ct.get_class_from_id(id).label
        return 'N/A'

    def get_ancestors_adjusted_frequencies(self, frequencies: Dict[str, float]):
        """
        Adjusts frequencies to favor more general concepts first (generic term will probably
        have less ancestors). Modifies the score according to the following formula :
        new_score = 0.75 * old_score - nb_ancestors * 0.25
        """
        adjusted_frequencies = dict()
        for elem in frequencies.items():
            id, count = elem

            if id not in self.snomed_ct.id_to_classes or id in self.full_exclude_ids:
                continue
            else:
                # We want to favor more general concepts first 
                # A generic term will probably have less ancestors
                ancestors = self.snomed_ct.get_ancestors_of_id(id, return_list=True)
                nb_ancestors = max(1, len(ancestors))
                new_score = int(10 * (0.75 * count - nb_ancestors * 0.25))
                adjusted_frequencies[id] = new_score
        return adjusted_frequencies

    def get_most_frequent_concepts(self, clinical_note, top_n):
        
        snomed_ids = self.annotator.annotate(clinical_note, return_ids_only=True) 
        all_ids = self.get_all_ancestors_and_ids(snomed_ids)

        results = Counter(all_ids)
        
        adjusted_frequencies = self.get_ancestors_adjusted_frequencies(results)

        if top_n == -1:
            top_n = len(adjusted_frequencies)
        
        adj_results = Counter(adjusted_frequencies).most_common(top_n)
        return list(map(lambda x: x[0], adj_results))
    
    def start_multiple(
        self, 
        notes: list, 
        top_n=5, 
        batch_size=1, 
        debug=0, 
        generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig(),
    ):
        """
        Starts the prompting process of multiple clinical notes

        Args
            - notes             : Clinical notes to process
            - top_n             : Number of concepts that will be extracted per notes (by frequency)
            - batch_size        : Number of concepts to process in parallel
            - debug             : Debug level
            - generation_config : Generation config used during inference
        """
        self.attributes.clear()
        self.attributes_by_id.clear()

        if debug:
            print('Starting')
        # for i, note in enumerate(notes):
        for i, note in tqdm(enumerate(notes), total=len(notes)):
            self.current_note_id = i
            self.attributes_by_id.append({})
            self.attributes.append({})
            self.start(
                note, 
                top_n=top_n, 
                batch_size=batch_size, 
                debug=debug, 
                generation_config=generation_config
            )
    
    
    def start(
        self, 
        clinical_note, 
        top_n=5, 
        batch_size=1, 
        debug=0, 
        generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()
    ):
        """
        Starts the prompting process of a clinical note

        Args
            - clinical_note     : Clinical note to process
            - top_n             : Number of concepts that will be extracted per notes (by frequency)
            - batch_size        : Number of concepts to process in parallel
            - debug             : Debug level
            - generation_config : Generation config used during inference
        """
        stack = []
        stack.append(self.snomed_ct.base_class.id)

        most_frequent_concepts = self.get_most_frequent_concepts(clinical_note, top_n=top_n)
        if len(most_frequent_concepts) == 0:
            return
        if debug:
            print('Number of concepts extracted : ', len(most_frequent_concepts))
            print('Most frequent concepts : ', list(map(lambda x: x.label, self.snomed_ct.convert_ids_to_classes(most_frequent_concepts))))
        
        iteration = 1
        while len(stack) > 0:            
            start = max(0, (iteration - 1) * batch_size)
            end = min(len(most_frequent_concepts), iteration * batch_size)
            current_node_ids = most_frequent_concepts[start:end]
            self.extract_attribute(clinical_note, current_node_ids, debug=debug, generation_config=generation_config)
            iteration += 1
            if iteration * batch_size > len(most_frequent_concepts):
                break
    
    def create_property_sentence(self, node_id: str, node_label: str):
        """
        Creates the property sentence used in the prompt template for a given node
        """
        properties = self.snomed_ct.get_properties_of_id(node_id)
        if len(properties) == 0:
            return ''
        else:
            current_property_knowledge = '\n- '.join(map(lambda x: x.get_value(), properties))
            property_sentence = '' if len(current_property_knowledge.strip()) == 0 else f'{node_label} is characterized by : \n- {current_property_knowledge}\n'
            return property_sentence

    def create_prompts(self, clinical_note, current_node_ids: List[str], debug=0):
        """
        Creates the final prompts that will be used by the model. One prompt is created by concepts

        Args
            - clinical_note     : Clinical note
            - current_node_ids  : Concept ids in the ontology that will be used to create prompts
            - debug             : Debug level
        """
        prompts = []
        for node_id in current_node_ids:
            label = self.snomed_ct.get_class_from_id(node_id).label
            if self.use_properties:
                properties = self.create_property_sentence(node_id, label)
            else:
                properties = ''
            
            if len(label.strip()) > 0:
                prompt = self.template.question_template.format_map({
                    'clinical_note': clinical_note,
                    'label': label,
                    'properties': properties
                })

                if debug:
                    print(f'\n{Color.CYAN}[Asking]{Color.OFF} : ', self.template.question_template.format_map({
                        'clinical_note': 'clinical note',
                        'label': label,
                        'properties': properties
                    }))
                
                prompts.append(prompt)
        return prompts
        
    def extract_attribute(
        self, 
        clinical_note: str, 
        current_node_ids: List[str], 
        debug=0,
        generation_config: DiverseBeamSearchConfig = DiverseBeamSearchConfig()
    ):
        """
        Extract attributes from a clinical note according to given concepts in ontology

        Args
            - clinical_note     : Clinical note
            - current_node_ids  : Ids of concepts in the ontology
            - debug             : Debug level
            - generation_config : Generation config to be used during inference
        """
        # We don't want to ask questions about the base class
        current_node_ids = list(filter(lambda x: x != self.snomed_ct.base_class.id, current_node_ids))

        if len(current_node_ids) == 0:
            return

        # Asking the model
        prompts = self.create_prompts(clinical_note, current_node_ids, debug)
        answers = self.constrained_model.ask(prompts, current_node_ids, generation_config)
        if debug:
            for answer in answers:
                # debug_answer = answer[:50] if len(answer) > 0 else answer
                print(f'\n{Color.RED}[Answer]{Color.OFF} : ', answer.strip())
        
        # Storing answers
        self.store_answers(current_node_ids, answers)

    def store_answers(self, node_ids: List[str], answers: List[str]):
        """
        Interprets answers of model about a set of ontology concepts

        Args :
            - node_ids  : Ids of concepts that were used to create the prompts associated with the `answers`
            - answers   : Answers to the prompts created by the ids of concepts in `node_ids`
        """
        if len(node_ids) != len(answers):
            raise ValueError(f'Length of the questions ({len(node_ids)}) should be the same as the length of the answers ({len(answers)})')

        for node_id, answer in zip(node_ids, answers):
            valid_answer = 'N/A' not in answer.strip()
            label = self.snomed_ct.get_class_from_id(node_id).label
            if len(answer.strip()) > 0 and valid_answer:
                self.attributes[self.current_note_id][label] = answer
                self.attributes_by_id[self.current_note_id][node_id] = answer
            else:
                self.attributes[self.current_note_id][label] = 'N/A'
                self.attributes_by_id[self.current_note_id][node_id] = 'N/A'


class OntologyBasedAnalyzer:
    """
    Responsable for the pruning step
    """
    
    def __init__(
        self, 
        result, 
        annotator: Annotator, 
        snomed: SNOMED, 
        tokenizer,
        notes_column: str = 'notes',
        attributes_column: str = 'attributes',
        attributes_by_id_column: str = 'attributes_by_id'
    ):
        self.result = result
        self.annotator = annotator
        self.snomed = snomed
        self.tokenizer = tokenizer
        
        self.notes_column = notes_column
        self.attributes_column = attributes_column
        self.attributes_by_id_column = attributes_by_id_column

        self.parse_data()
        
    def parse_data(self):
        """
        Parses the data given to transform an admission note to multiple clinical notes and analyze
        the extracted values of these notes
        """
        self.raw_notes = self.result.notes
        self.notes = preprocess_clinical_notes(self.result[self.notes_column], self.tokenizer)
        
        self.attributes = ast.literal_eval(self.result[self.attributes_column])
        self.attributes_by_id = ast.literal_eval(self.result[self.attributes_by_id_column])
        self.exclude_ids = set([self.snomed.base_class.id, '362981000', '123037004', '276339004', '106237007', '444677008']) # Linkage concepts, qualifier value, etc

        self.preprocess_data()

    def preprocess_data(self):
        """
        Processes the answers given by the model for better formatting. 
        """
        for i, attr in enumerate(self.attributes):            
            for k, v in attr.items():
                real_value = v.replace('. \n', '')
                real_value = real_value.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                # Detect N/A values
                if real_value.find('The clinical note does not') >= 0:
                    real_value = 'N/A'
                attr[k] = real_value

        for i, attr in enumerate(self.attributes_by_id):            
            for k, v in attr.items():
                real_value = v.replace('. \n', '')
                real_value = real_value.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                # Detect N/A values
                if real_value.find('The clinical note does not') >= 0:
                    real_value = 'N/A'
                attr[k] = real_value


    def show_results(self, attributes = None, show_na = False):
        """
        Shows the results in a formatted way. If `attributes` is None, `self.attributes` will
        be shown

        Args 
            - attributes    : Attributes to show
            - show_na       : Whether to show N/A values or not
        """

        if attributes is None:
            attributes = self.attributes
            
        if isinstance(attributes, dict):
            for k, v in attributes.items():
                    real_value = v.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                    can_show = show_na or (not show_na and 'N/A' not in real_value)
                    can_show &= len(real_value) > 0 
                    if can_show:
                        print(f'\x1b[31m{k}\x1b[0m', ' : ', real_value)
            return
        
        for i, attr in enumerate(attributes):            
            if len(attr) > 0:
                print('=' * 10, f'Note {i}', '=' * 10)
            for k, v in attr.items():
                real_value = v.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                can_show = show_na or (not show_na and 'N/A' not in real_value)
                can_show &= len(real_value) > 0 
                if can_show:
                    print(f'\x1b[31m{k}\x1b[0m', ' : ', real_value)

    def get_printable_results(self, attributes = None, show_na = False):
        """
        Returns the results in a formatted way. If `attributes` is None, `self.attributes` will
        be shown

        Args 
            - attributes    : Attributes to show
            - show_na       : Whether to show N/A values or not

        Returns
        Formatted attributes in a string
        """
        if attributes is None:
            attributes = self.attributes
        
        out = ''
        if isinstance(attributes, dict):
            for k, v in attributes.items():
                real_value = v.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                can_show = show_na or (not show_na and 'N/A' not in real_value)
                can_show &= len(real_value) > 0 
                if can_show:
                    out += k + ':' + real_value + '\n'
            return out
        

        for i, attr in enumerate(attributes):
            if len(attr) > 0:
                out += '=' * 10 + f' Note {i} ' + '=' * 10 + '\n'
            for k, v in attr.items():
                real_value = v.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                can_show = show_na or (not show_na and 'N/A' not in real_value)
                can_show &= len(real_value) > 0 
                if can_show:
                    out += k + ':' + real_value + '\n'
        return out
    
    def filter_results(self, x: UMLSAnnotatorResult):
        """
        Removes results containing excluded branches of SNOMED
        """
        ancestors = self.snomed.get_ancestors_of_id(x.snomed_id, return_set=True)
        ancestors.add(x.snomed_id)
        return len(ancestors.intersection(self.exclude_ids)) == 1 # Only the base class

    def get_filtered_attributes(self, attributes: Dict[str, str], filter_attributes: List[str], ancestor_level=2):
        """
        Computes the intersection of two attribute list by taking all attributes from `attributes` 
        that are are least an `ancestor_level` distance from attributes in `filter_attributes`.
        """
        # print('Number of attributes extracted : ', len(attributes))
        filtered_attributes = dict()
        for id, value in attributes.items():
            ancestors_id: set = self.snomed.get_ancestors_of_id(id, return_set=True)
            for filtered in filter_attributes:
                ancestors: set = self.snomed.get_ancestors_of_id(filtered, return_set=True)
                if len(ancestors_id.intersection(ancestors)) >= ancestor_level:
                    # Two classes are related if they have more than `ancestor_level` common ancestors
                    filtered_attributes[id] = value
        # print('Number of attributes kept : ', len(filtered_attributes))
        return filtered_attributes

    def get_adjusted_frequencies(self, initial_frequencies: Dict[str, str]):
        """
        Adjusts the frequencies of a frequency dictionary by taking into account the number of ancestor
        to favor more precise concepts
        """
        adjusted_frequencies = dict()
        for elem in initial_frequencies.items():
            id, count = elem

            if id not in self.snomed.id_to_classes or id in self.exclude_ids:
                continue
            else:
                # We want to favor more general concepts in this case 
                # A generic term will probably have less ancestors
                ancestors = self.snomed.get_ancestors_of_id(id, return_list=True)
                nb_ancestors = max(1, len(ancestors))
                
                new_score = 0.95 * count - 0.05 * nb_ancestors
                adjusted_frequencies[id] = new_score
        return adjusted_frequencies


    def counter_list_to_dict(self, tuple_list: List):
        """
        Converts a list of tuples in the form `(elem, count)` to a dictionary in the form `dict[elem] = count`
        """
        filtered_most_common = dict()
        for common in tuple_list:
            id, count = common
            # if id not in self.exclude_ids:
            filtered_most_common[id] = count
        return filtered_most_common

    def adapt_to_domain(
        self, 
        domain_frequencies: DomainClassFrequency,
        top_n: int = 50, 
        ancestor_level: int = 1, 
        adjust_frequencies: bool = True,
    ):
        """
        Pruning phase.

        Args
            - domain_frequencies    : Dictionary containing the frequencies of concepts associated to a certain domain 
            - top_n                 : Number of concepts that should be taken into account in pruning (in order of frequency)
            - ancestor_level        : Number of hops between a valid node and the current node in the ontology
            - adjust_frequencies    : Whether to adjust the frequencies according to their ancestors (to favor specific concepts)
        """
        if adjust_frequencies:
            domain_frequencies.frequencies = self.get_adjusted_frequencies(domain_frequencies.frequencies)

        if top_n == -1:
            top_n = len(domain_frequencies.frequencies)

        filtered_most_common = domain_frequencies.get_most_common(top_n=top_n, exclude_set=self.exclude_ids, snomed=self.snomed)
        filtered_most_common = [x[0] for x in filtered_most_common] # [(id1, count1), (id2, count2), ...] -> [id1, id2, ...]
        final_domain_attributes = []
        for attribute in self.attributes_by_id:
            domain_attributes = self.get_filtered_attributes(attribute, filtered_most_common, ancestor_level=ancestor_level)
            final_domain_attributes.append(domain_attributes)
        
        return list(map(lambda x: self.convert_ids_to_labels(x), final_domain_attributes)), final_domain_attributes
    
    def convert_ids_to_labels(self, attributes):
        """
        Converts a list of ids in the ontology to a list of labels
        """
        labeled_attributes = {}
        for k, v in attributes.items():
            labeled_attributes[self.snomed.get_class_from_id(k).label] = v
        return labeled_attributes


class BHCOntologyBasedVerbalizer:
    """
    Applies the pruning + verbalizer stage to an unstructured input for the BHC task
    """
    def __init__(
        self, 
        output_path: str,
        snomed: SNOMED,
        annotator: Annotator,
        model,
        tokenizer,
    ) -> None:
        """
        Args
            - output_path   : Path where to save the results
            - snomed        : SNOMED ontology
            - annotator     : Annotator that can tag concepts in a text
            - model         : Model to use for inference
            - tokenizer     : Tokenizer linked to the model
        """

        self.output_path = output_path
        self.snomed = snomed
        self.annotator = annotator
        self.model = model
        self.tokenizer = tokenizer

        if os.path.exists(self.output_path):
            self.df = pd.read_csv(self.output_path)
        else:
            self.df = pd.DataFrame(
                [], 
                columns=[
                    'text', 
                    'summary', 
                    'structured', 
                    'unstructured',
                    'prompt'
                ]
            )

    def ask_model_batch(self, x: list[str], max_new_tokens=1536, limit_tokens = 14000):
        """
        Batched inference on model

        Args
            - x                 : Input to send to model
            - max_new_tokens    : Max number of tokens to generate
            - limit_tokens      : Max number of tokens to consider. Otherwise, input is not sent to model
        """
        clear_gpu_cache()
        chat_template = ChatTemplate(self.tokenizer)    
        prompts = list(map(lambda y: chat_template.add_user_entry(y), x))        
        model_input = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        if limit_tokens > 0 and model_input['input_ids'].shape[-1] >= limit_tokens:
            return ''
        
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **model_input, 
                max_new_tokens=max_new_tokens,
            )
            # Only retrieve the newly generated tokens
            new_tokens = generated[:, model_input['input_ids'].shape[-1]:]

            results = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            del model_input
            return results


    def start(
        self,
        structured: pd.DataFrame,
        pre_prompt: str,
        post_prompt: str,
        extract: bool = True,
        prune: bool = False,
        domain_frequencies: DomainClassFrequency = None,
        top_n_concepts: int = 100,
        ancestor_level: int = 2,
        text_column: str = 'text',
        summary_column: str = 'summary',
        debug: int = 0,
    ):
        """
        Applies the pruning + verbalizer step to a set of inputs

        Args 
            - structured            : DataFrame containing a structured representation of the input (in columns attributes and attributes_by_id)
            - pre_prompt            : Text to add before the structured representation of the input
            - post_prompt           : Text to add after the structured representation of the input
            - extract               : Whether to extract or not (if not, simple inference to summarize the notes)
            - prune                 : Whether to apply the pruning step or not
            - domain_frequencies    : Frequencies of domain for the pruning step
            - top_n                 : Number of concepts that should be taken into account in pruning (in order of frequency)
            - ancestor_level        : Number of hops between a valid node and the current node in the ontology
            - text_column           : Name of column containing the clinical notes
            - summary_column        : Name of column containing the summary (BHC)
            - debug                 : Debug level
        """
        initial = len(self.df)
        for i in tqdm(range(initial, len(structured)), total=len(structured), initial=initial):
            
            current_result = structured.iloc[i]
            analyzer = OntologyBasedAnalyzer(
                current_result, 
                self.annotator, 
                self.snomed, 
                self.tokenizer,
                notes_column='notes',
                attributes_column='attributes',
                attributes_by_id_column='attributes_by_id'
            )
            id = current_result['id']
            summary = current_result[summary_column]
            text = current_result[text_column]

            if not extract:
                # Normal generation
                prompt = pre_prompt + text + post_prompt
            elif extract and not prune:
                # Extraction + Verbalizer
                prompt = analyzer.get_printable_results()
                prompt = pre_prompt + prompt + post_prompt
            else:
                # Extraction + Pruning + Verbalizer
                domain, _ = analyzer.adapt_to_domain(
                    domain_frequencies, 
                    top_n=top_n_concepts, 
                    ancestor_level=ancestor_level
                )
                prompt = analyzer.get_printable_results(domain)
                prompt = pre_prompt + prompt + post_prompt
            
            if debug >= 2:
                print(prompt)
            
            unstructured = self.ask_model_batch([prompt])
            if debug >= 1:
                print(unstructured)

            new_row = {
                'id': id,
                'text': text, 
                'summary': summary,
                'structured': current_result.attributes_by_id,
                'unstructured': unstructured,
                'prompt': prompt
            }

            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            self.df.to_csv(self.output_path)

class DomainOntologyBasedVerbalizer:
    """
    Applies the pruning + verbalizer stage to an unstructured input
    """

    DOMAIN_HEADER = '<|domain|>'

    def __init__(
        self, 
        output_path: str,
        domain_analysis: Dict[str, DomainClassFrequency],
        snomed: SNOMED,
        annotator: Annotator,
        model,
        tokenizer,
    ) -> None:
        self.output_path = output_path
        self.domain_analysis = domain_analysis
        self.snomed = snomed
        self.annotator = annotator
        self.model = model
        self.tokenizer = tokenizer

        if os.path.exists(self.output_path):
            self.df = pd.read_csv(self.output_path)
        else:
            self.df = pd.DataFrame(
                [], 
                columns=[
                    'text', 
                    'summary', 
                    'structured', 
                    'unstructured',
                    'domain'
                ]
            )

    def ask_model_batch(self, x: list[str], max_new_tokens=1536):
        """
        Runs inference on the verbalizer

        Args
            - x                 : input to send to the verbalizer
            - max_new_tokens    : Max number of tokens to be newly generated
        
        """
        clear_gpu_cache()
        chat_template = ChatTemplate(self.tokenizer)    
        prompts = list(map(lambda y: chat_template.add_user_entry(y), x))        
        model_input = self.tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **model_input, 
                max_new_tokens=max_new_tokens,
            )
            # Only retrieve the newly generated tokens
            new_tokens = generated[:, model_input['input_ids'].shape[-1]:]

            results = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            del model_input
            return results


    def start(
        self,
        structured: pd.DataFrame,
        pre_prompt: str,
        post_prompt: str,
        normal: bool = True,
        top_n_concepts: int = 100,
        ancestor_level: int = 4,
        text_column: str = 'text',
        summary_column: str = 'summary',
        domain_filter: List[str] = None
    ):
        """
        Applies the pruning + verbalizer step to a set of inputs

        Args 
            - structured            : DataFrame containing a structured representation of the input (in columns attributes and attributes_by_id)
            - pre_prompt            : Text to add before the structured representation of the input
            - post_prompt           : Text to add after the structured representation of the input
            - extract               : Whether to extract or not (if not, simple inference to summarize the notes)
            - prune                 : Whether to apply the pruning step or not
            - domain_frequencies    : Frequencies of domain for the pruning step
            - top_n                 : Number of concepts that should be taken into account in pruning (in order of frequency)
            - ancestor_level        : Number of hops between a valid node and the current node in the ontology
            - text_column           : Name of column containing the clinical notes
            - summary_column        : Name of column containing the summary (BHC)
            - debug                 : Debug level
        """
        initial = len(self.df) // len(self.domain_analysis)
        for i in tqdm(range(initial, len(structured)), total=len(structured), initial=initial):
            
            current_result = structured.iloc[i]
            analyzer = OntologyBasedAnalyzer(
                current_result, 
                self.annotator, 
                self.snomed, 
                self.tokenizer,
                notes_column='notes',
                attributes_column='attributes',
                attributes_by_id_column='attributes_by_id'
            )
            summary = current_result[summary_column]
            text = current_result[text_column]

            for domain, dcf in self.domain_analysis.items():

                if domain_filter is not None and domain not in domain_filter:
                    continue
                print(f'Adapting to {domain} domain')

                if normal:
                    prompt = pre_prompt + text + post_prompt
                else:
                    domain_results, _ = analyzer.adapt_to_domain(
                        dcf,
                        top_n=top_n_concepts, 
                        ancestor_level=ancestor_level
                    )
                    prompt = analyzer.get_printable_results(domain_results)
                    prompt = pre_prompt + prompt + post_prompt
                
                prompt = prompt.replace(DomainOntologyBasedVerbalizer.DOMAIN_HEADER, domain)
                unstructured = self.ask_model_batch([prompt])

                new_row = {
                    'text': text, 
                    'summary': summary,
                    'structured': current_result.attributes_by_id,
                    'unstructured': unstructured,
                    'domain': domain
                }

                self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
                # self.df_results = pd.DataFrame(predictions, columns=['clinical_notes', 'aces_prediction', 'aces_bhc_prediction', 'normal_prediction', 'target'])
                self.df.to_csv(self.output_path)
