
import sqlite3
from typing import List
from abc import ABC, abstractmethod

from quickumls import QuickUMLS
from medcat.cat import CAT

from src.ontology.snomed import SNOMED

import spacy
from spacy import displacy

class Match:
    """
    Match by annotator (only SNOMED is used in this case)
    """
    def __init__(self, dict_match) -> None:
        self.start = dict_match['start']
        self.end = dict_match['end']
        self.ngram = dict_match['ngram']
        self.term = dict_match['term']
        self.cui = dict_match['cui']
        self.similarity = dict_match['similarity']
        self.semtypes = dict_match['semtypes']
        self.preferred = dict_match['preferred']

class AnnotatorResult:
    """
    Result of Annotator based on a match
    """

    def __init__(self, match: Match, semantic_types: set[str] = [], snomed_id: str = '') -> None:
        self.match = match
        self.concept = self.match.term
        self.cui = self.match.cui
        self.snomed_id = snomed_id
        self.semantic_types = semantic_types
        self.semantic_types_ids = self.match.semtypes

    def __str__(self) -> str:
        return f'{self.concept} (SEMANTIC TYPES : {self.semantic_types}, SNOMED_ID : "{self.snomed_id}")'

class UMLSDatabase:
    """ UMLS lookup database """
    
    def __init__(self, path: str):
        """
        Args
            - path: Path to the .db file 
        """
        self.sqlite = sqlite3.connect(path)
    
    def list_tables(self):
        """ Lists the tables """
        return self.sqlite.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    def get_columns_of_table(self, table_name: str):
        """ 
        Lists the columns of a table
        
        Args
            - table_name    : Name of table
        """
        return self.sqlite.execute(f"PRAGMA table_info({table_name});").fetchall()

    def head_of_table(self, table_name: str):
        """ 
        Returns the 5 first elements of a table
        
        Args
            - table_name    : Name of table
        """
        return self.sqlite.execute(f"SELECT * FROM {table_name} LIMIT 5;").fetchall()

    def execute(self, query: str, *args):
        """
        Executes a query on the database

        Args
            - query : Query to execute
        """
        return self.sqlite.execute(query, *args)

    def get_semantic_types_of_cui(self, cui: str):
        """
        Returns the semantic types of a cui (ID)
        """
        return list(map(lambda x: x[0], self.execute(f"SELECT STY FROM MRSTY WHERE CUI = ?", (cui,)).fetchall()))

    def get_all_semantic_types(self):
        """
        Returns all semantic types in database
        """
        return list(map(lambda x: x[0], self.execute("SELECT DISTINCT STY FROM MRSTY")))

    def get_information_of_cui(self, cui: str, snomed_only=True):
        """
        Returns the information of a cui (label, semantic types, etc)
        """
        query = "SELECT MRSTY.CUI, MRCONSO.STR, MRSTY.STY, MRCONSO.SCUI FROM MRSTY INNER JOIN MRCONSO ON MRSTY.CUI = MRCONSO.CUI WHERE MRSTY.CUI = ? AND MRCONSO.SAB='SNOMEDCT_US'"
        if not snomed_only:
            query = "SELECT MRSTY.CUI, MRCONSO.STR, MRSTY.STY, MRCONSO.SCUI FROM MRSTY INNER JOIN MRCONSO ON MRSTY.CUI = MRCONSO.CUI WHERE MRSTY.CUI = ?"
        return list(self.execute(query, (cui,)).fetchall())
    
    def get_snomed_id(self, cui: str):
        """
        Gets the snomed id of a cui
        """
        query = "SELECT MRCONSO.SCUI FROM MRSTY INNER JOIN MRCONSO ON MRSTY.CUI = MRCONSO.CUI WHERE MRSTY.CUI = ? AND MRCONSO.SAB='SNOMEDCT_US'"
        return list(map(lambda x: x[0], self.execute(query, (cui,)).fetchall()))
    
    def get_term(self, cui: str):
        """
        Gets the label of a cui
        """
        return self.execute("SELECT STR FROM MRCONSO WHERE MRCONSO.CUI = ?", (cui,))

    
class Annotator(ABC):
    """
    Abstract class for an annotator
    """

    @abstractmethod
    def annotate(self, text: str, snomed_only = True, return_raw_matches = False, return_ids_only = False, result_filter = None) -> List:
        """
        Annotates a text

        Args
            - text                  : Text to annotate
            - snomed_only           : Whether to retrieve only snomed concepts
            - return_raw_matches    : Whether to return the raw matches or results from AnnotatorResult
            - return_ids_only       : Whether to return only the ids of the matches
            - result_filter         : Function to pre-filter the results

        Returns
        Annotations of text
        """
        pass

    def batch_annotate(self, texts: List[str], snomed_only = True, return_raw_matches = False, return_ids_only = False, result_filter = None) -> List:
        """
        Annotates a text

        Args
            - text                  : Text to annotate
            - snomed_only           : Whether to retrieve only snomed concepts
            - return_raw_matches    : Whether to return the raw matches or results from AnnotatorResult
            - return_ids_only       : Whether to return only the ids of the matches
            - result_filter         : Function to pre-filter the results

        Returns
        Annotations of texts
        """
        results = []
        for text in texts:
            results.append(self.annotate(text, snomed_only=snomed_only, return_raw_matches=return_raw_matches, return_ids_only=return_ids_only))
        return results

    @abstractmethod
    def render(self, text: str, snomed_only = True, render_snomed_ids = False, result_filter = None):
        """
        Renders the annotations of a text

        Args
            - text                      : Text to annotate
            - snomed_only               : Whether to retrieve only snomed concepts
            - return_snomed_ids_only    : Whether to render the snomed ids only
            - result_filter             : Function to pre-filter the results

        """
        pass

class MedCatAnnotator(Annotator):
    """
    MedCat Annotator wrapper
    """

    def __init__(self, medcat_path: str, snomed: SNOMED, device: str = None, meta_cat_config_dict = None) -> None:
        self.path = medcat_path
        if device and meta_cat_config_dict is None:
            config={
                'general': {
                    'device': device
                }
            }
        else:
            config = meta_cat_config_dict

        self.cat = CAT.load_model_pack(medcat_path, meta_cat_config_dict=config)
        
        self.snomed = snomed
        
    def process_entities(self, entities):
        results = []
        for k, v in entities['entities'].items():
            dict_match = {}
            dict_match['start'] = v['start']
            dict_match['end'] = v['end']
            dict_match['ngram'] = ''
            dict_match['term'] = v['detected_name']
            dict_match['cui'] = v['cui']
            dict_match['similarity'] = v['context_similarity']
            dict_match['semtypes'] = v['type_ids']
            dict_match['preferred'] = v['pretty_name']
            
            match = Match(dict_match)
            result = AnnotatorResult(match, v['types'], v['cui'])
            results.append(result)
        
        return results

    def annotate(self, text, snomed_only = True, return_raw_matches = False, return_ids_only = False, result_filter = None):
        """
        Annotates a text

        Args
            - text                  : Text to annotate
            - snomed_only           : Whether to retrieve only snomed concepts
            - return_raw_matches    : Whether to return the raw matches or results from AnnotatorResult
            - return_ids_only       : Whether to return only the ids of the matches
            - result_filter         : Function to pre-filter the results

        Returns
        Annotations of text
        """
        ents = self.cat.get_entities(text, only_cui=return_ids_only)

        if return_ids_only:
            return list(ents['entities'].values())
        
        results = self.process_entities(ents)
        if result_filter is not None:
            results = list(filter(result_filter, results))

        if return_raw_matches:
            return results, list(ents['entities'].values())
        
        return results
    
    def batch_annotate(self, texts: List[str], snomed_only=True, return_raw_matches=False, return_ids_only=False, result_filter = None) -> List:
        """
        Annotates a text

        Args
            - text                  : Text to annotate
            - snomed_only           : Whether to retrieve only snomed concepts
            - return_raw_matches    : Whether to return the raw matches or results from AnnotatorResult
            - return_ids_only       : Whether to return only the ids of the matches
            - result_filter         : Function to pre-filter the results

        Returns
        Annotations of texts
        """
        results = self.cat.get_entities_multi_texts(texts, only_cui=return_ids_only)
        if return_ids_only:
            return list(map(lambda x: list(x['entities'].values()), results))

        processed_results = list(map(self.process_entities, results))
        if result_filter is not None:
            processed_results = list(filter(result_filter, processed_results))

        if return_raw_matches:
            return processed_results, results
        return processed_results

    def render(self, text, snomed_only = True, render_snomed_ids=False, result_filter = None):
        """
        Renders the annotations of a text

        Args
            - text                      : Text to annotate
            - snomed_only               : Whether to retrieve only snomed concepts
            - return_snomed_ids_only    : Whether to render the snomed ids only
            - result_filter             : Function to pre-filter the results

        """
        nlp = spacy.blank('en')
        doc = nlp.make_doc(text)
        results = self.annotate(text, snomed_only, result_filter=result_filter)
        ents = []
        for result in results:
            if render_snomed_ids:
                if result.snomed_id in self.snomed.id_to_classes:
                    label = self.snomed.get_class_from_id(result.snomed_id).label
                else:
                    label = 'N/A'
                ent = doc.char_span(int(result.match.start), int(result.match.end), label=label)
            else:
                ent = doc.char_span(int(result.match.start), int(result.match.end), label=', '.join(result.semantic_types))
                
            if ent is not None:
                ents.append(ent)
        doc.ents = ents
        return displacy.render(doc, style='ent')

class UMLSAnnotator(Annotator):
    """
    Quick UMLS Annotator 
    """

    def __init__(self, umls_path: str, umls_db_path: str, threshold=0.7, window=5) -> None:
        """
        Args
            - umls_path     : Path to umls annotator
            - umls_db_path  : Path to SQLite database
            - threshold     : Threshold of similarity
            - window        : Window size for the matcher
        """
        self.matcher = QuickUMLS(umls_path, threshold=threshold, window=window, min_match_length=2)
        self.umls_db = UMLSDatabase(umls_db_path)

    def annotate(self, text, snomed_only = True, return_raw_matches = False, return_ids_only = False, result_filter = None):
        """
        Annotates a text

        Args
            - text                  : Text to annotate
            - snomed_only           : Whether to retrieve only snomed concepts
            - return_raw_matches    : Whether to return the raw matches or results from AnnotatorResult
            - return_ids_only       : Whether to return only the ids of the matches
            - result_filter         : Function to pre-filter the results

        Returns
        Annotations of text
        """
        matches = self.matcher.match(text, best_match=True, ignore_syntax=False)
        results = []
        for match in matches:
            for variant in match:
                umls_match = Match(variant)
                data = self.umls_db.get_information_of_cui(umls_match.cui, snomed_only=snomed_only)
                
                if len(data) == 0:
                    continue
                
                cui, concept, semantic_type, snomedct_id = data[0]
                # Data is a tuple containing the following values : (cui, concept, semantic type, snomed_id)
                
                semantic_types = set({d[2] for d in data})

                if snomed_only:
                    if len(snomedct_id) > 0:
                        results.append(AnnotatorResult(umls_match, semantic_types, snomedct_id))
                        break
                else:
                    results.append(AnnotatorResult(umls_match, semantic_types))
                    
        if return_raw_matches:
            return results, matches
        else:
            return results

    def batch_annotate(self, texts: List[str], snomed_only=True, return_raw_matches=False, return_ids_only=False, result_filter = None) -> List:
        """
        Annotates a text

        Args
            - text                  : Text to annotate
            - snomed_only           : Whether to retrieve only snomed concepts
            - return_raw_matches    : Whether to return the raw matches or results from AnnotatorResult
            - return_ids_only       : Whether to return only the ids of the matches
            - result_filter         : Function to pre-filter the results

        Returns
        Annotations of texts
        """
        return super().batch_annotate(texts, snomed_only, return_raw_matches, return_ids_only)


    def render(self, text, snomed_only = True, render_snomed_ids=False, result_filter = None):
        """
        Renders the annotations of a text

        Args
            - text                      : Text to annotate
            - snomed_only               : Whether to retrieve only snomed concepts
            - return_snomed_ids_only    : Whether to render the snomed ids only
            - result_filter             : Function to pre-filter the results

        """
        nlp = spacy.blank('en')
        doc = nlp.make_doc(text)
        results = self.annotate(text, snomed_only, result_filter=result_filter)
        ents = []
        for result in results:
            if render_snomed_ids:
                ent = doc.char_span(result.match.start, result.match.end, label=result.snomed_id)
            else:
                ent = doc.char_span(result.match.start, result.match.end, label=', '.join(result.semantic_types))
            ents.append(ent)
        doc.ents = ents
        return displacy.render(doc, style='ent')


def min_length_filter(minimum: int):
    return lambda x: len(x.concept) >= minimum

def max_length_filter(maximum: int):
    return lambda x: len(x.concept) <= maximum
