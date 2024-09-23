import re
from typing import List, Callable

from src.ontology.snomed import SNOMED
from src.ontology.umls import Annotator, AnnotatorResult

MAX_TOKENS = 2048
MIN_TOKENS = 64

class NoteProcessor:
    """
    Applies a series of functions to a list of note. The functions are expected
    to take a list of notes as an input and return a list of notes
    """

    def __init__(self, functions: List[Callable]) -> None:
        self.functions = functions

    def __call__(self, notes: List[str]) -> List[str]:
        """
        Transforms a list of notes based on the given functions

        Args :
            - notes : List of notes to transform

        Returns
        Transformed notes
        """
        if len(notes) == 0:
            return []
        
        new_notes = []
        for note in notes:
            new_note = note
            for function in self.functions:
                new_note = function(note)
            new_notes.append(new_note)
        return new_notes

class ClinicalNoteProcessor:
    """
    Class used to transform an admission note to a series of processed clinical notes. A processed clinical
    note does not contain abreviations (HR, rr, etc), contains less than a fixed number of tokens and does
    not contain any deintification tokens
    
    """

    ATTRIBUTE_NAMES = {
        'Temperature': ['T', 'Temp', 'Tc', 'Tcurrent', 'T current'],
        'Blood pressure': ['bp', 'BP'],
        'Heart rate': ['hr', 'HR', 'p'],
        'Respiratory rate': ['rr', 'r'],
        'Oxygen saturation': ['saO2', 'o2', 'saturation', 'o2sat', '02sat'],
    }

    def __init__(self, tokenizer, max_length = 2048, max_nb_notes = 20) -> None:
        """
        Args :
            - tokenizer     : Tokenizer to use to find the length of a note
            - max_length    : Maximal number of tokens a note can have
            - max_nb_notes  : Maximal number of clinical notes in an admission note
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_nb_notes = max_nb_notes

    def remove_long_notes(self, notes: List[str]):
        for note in notes:
            length = len(self.tokenizer.tokenize(note))
            if length > self.max_length:
                return []
        return notes

    def replace_vitals_abreviations(self, note: str):
        """
        Replaces all abreviations related to vital signs in a note
        """
        new_note = note
        for attribute, aliases in ClinicalNoteProcessor.ATTRIBUTE_NAMES.items():
            for alias in aliases:
                pattern = rf'({re.escape(alias)})((?: |:| |-|)(?: ?)(?:\d+(?:\/| |)\d+%?))'
                new_note = re.sub(pattern, lambda x: f"{attribute}{x.group(2)}", new_note, re.IGNORECASE)
        return new_note

    def admission_to_clinical_notes(self, admission_note: str) -> List[str]:
        """
        Remove all headers in an admission note that was added previously to merge the clinical notes.
        The separator '==========' is used to split the admission note into multiple clinical notes. If
        an admission note contains more than `self.max_nb_notes` clinical notes, an empty array is returned.

        Args :
            - admission_note : The admission note containing multiple clinical notes separated by the 
            header

        Returns
        List of individual notes of the admission. Empty if the admission note contains more than `self.max_nb_notes`.
        """
        clinical_notes = admission_note
        no_heading_notes = []
        
        splitted_notes = clinical_notes.split('==========')
        if len(splitted_notes) > self.max_nb_notes:
            return []
        
        for current_note in splitted_notes:
            no_heading = current_note.split('\n')
            
            if len(no_heading[0].strip()) > 0:
                no_heading = '\n'.join(no_heading[1:])
            else:
                no_heading = '\n'.join(no_heading[2:])
            
            no_heading_notes.append(no_heading)

        result = '\n==========\n'.join(no_heading_notes)
        result = result.replace('\n\n', '\n').replace('  ', ' ')
        result = ' '.join(result.split(sep=' '))
        return result.split('==========')


    def remove_deidentification_headers(self, note: str):
        """
        Removes all deintification headers in MIMIC-III
        """
        return re.sub(r'\[\*\*(.*?)\*\*\]', '', note)

    def __call__(self, admission_note: str) -> List[str]:
        """
        Transforms an admission note into a series of clinical notes
        """
        notes = self.admission_to_clinical_notes(admission_note)
        notes = self.remove_long_notes(notes)
        processed_notes = []
        for i, note in enumerate(notes):
            processed_note = self.replace_vitals_abreviations(note)
            processed_note = self.remove_deidentification_headers(processed_note)
            processed_notes.append(processed_note)
        return processed_notes

def replace_abreviations(text: str, annotator: Annotator, snomed: SNOMED):
    """
    Replaces all abreviations in a text by the associated label in the ontology.
    An abreviation is defined by every sequence of characters that can be tagged by
    the annotator to a concept in the SNOMED ontology, but that is not equivalent to
    the label in the ontology

    Args :
        - text      : Text containing abreviations
        - annotator : Annotator that can tag concepts in a text to the SNOMED ontology
        - snomed    : Ontology used to retrieve the concepts' labels

    Returns
    Text with all abreviations removed
    """
    def filter(res: AnnotatorResult):
        char_before = text[res.match.start - 1]
        char_after = text[res.match.end]
        t = text[res.match.start:res.match.end]
        # print(t, ' - ', char_before, ' - ', char_after)
        accepted = (not char_before.isalpha()) or \
            (not char_after.isalpha())
        return accepted

    annotations: List[AnnotatorResult] = annotator.annotate(text, result_filter=filter)

    # Sort annotations by their start position in ascending order
    sorted_annotations = sorted(annotations, key=lambda a: a.match.start)
    
    offset = 0
    for annotation in sorted_annotations:
        if annotation.snomed_id not in snomed.id_to_classes:
            continue

        start = annotation.match.start + offset
        end = annotation.match.end + offset
        current_term = text[start:end]
        sclass = snomed.get_class_from_id(annotation.snomed_id)
        
        if current_term.lower() == sclass.label.lower():
            continue

        replacement = f'{current_term} ({sclass.label})'
        text = text[:start] + replacement + text[end:]
        
        # Update offset
        offset += len(replacement) - (end - start)

    return text

def remove_incompatible_lengths(notes, tokenizer):
    """
    Removes all notes that do not contain between a MIN_TOKENS and MAX_TOKENS tokens
    """
    valid_notes = []
    for current_note in notes:
        token_length = len(tokenizer.tokenize(current_note))
        if token_length >= MIN_TOKENS and token_length <= MAX_TOKENS:
            valid_notes.append(current_note)
        # else:
            # print(f'Note was removed, because not in token range : {token_length}')
    return valid_notes

def remove_headers(clinical_notes):
    """
    Remove all headers in an admission note that was added previously to merge the clinical notes.
    The separator '==========' is used to split the admission note into multiple clinical notes. If
    an admission note contains more than `self.max_nb_notes` clinical notes, an empty array is returned.

    Args :
        - clinical_notes : The admission note containing multiple clinical notes separated by the 
        header

    Returns
    List of individual notes of the admission. Empty if the admission note contains more than `self.max_nb_notes`.
    """
    no_heading_notes = []
    
    splitted_notes = clinical_notes.split('==========')
    if len(splitted_notes) > 20:
        return []
    
    for current_note in splitted_notes:
        no_heading = current_note.split('\n')
        
        if len(no_heading[0].strip()) > 0:
            no_heading = '\n'.join(no_heading[1:])
        else:
            no_heading = '\n'.join(no_heading[2:])
        
        no_heading_notes.append(no_heading)

    result = '\n==========\n'.join(no_heading_notes)
    result = result.replace('\n\n', '\n').replace('  ', ' ')
    result = ' '.join(result.split(sep=' '))
    return result.split('==========')

def lower_all(notes: List[str]):
    """Lowers all clinical notes"""
    return list(map(lambda x: x.lower(), notes))

def remove_deidentification_headers(notes: List[str]):
    """
    Removes all deintification headers in MIMIC-III
    """
    cleaned_notes = []
    for note in notes:
        cleaned_note = re.sub(r'\[\*\*(.*?)\*\*\]', '', note)
        cleaned_notes.append(cleaned_note)
    return cleaned_notes

def remove_abreviations(notes):
    """
    Replaces all abreviations related to vital signs in a note
    """
    ATTRIBUTE_NAMES = {
        'Temperature': ['T', 'Temp', 'Tc', 'Tcurrent', 'T current'],
        'Blood pressure': ['bp', 'BP'],
        'Heart rate': ['hr', 'HR', 'p'],
        'Respiratory rate': ['rr', 'r'],
        'Oxygen saturation': ['saO2', 'o2', 'saturation', 'o2sat', '02sat'],
    }
    new_notes = []
    for note in notes:
        new_note = note
        for attribute, aliases in ATTRIBUTE_NAMES.items():
            for alias in aliases:
                pattern = rf'({re.escape(alias)})((?: |:| |-|)(?: ?)(?:\d+(?:\/| |)\d+%?))'
                new_note = re.sub(pattern, lambda x: f"{attribute}{x.group(2)}", new_note, re.IGNORECASE)
        new_notes.append(new_note)
    return new_notes

def preprocess_clinical_notes(notes, tokenizer):
    notes = remove_headers(notes)
    notes = remove_deidentification_headers(notes)
    notes = remove_abreviations(notes)
    notes = remove_incompatible_lengths(notes, tokenizer)
    return notes
