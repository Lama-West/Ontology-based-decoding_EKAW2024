from collections import Counter, defaultdict
from typing import Dict, List
from tqdm import tqdm
import joblib

from src.ontology.snomed import SNOMED
from src.ontology.umls import AnnotatorResult

MAX_ANCESTORS = 2

class DomainClassFrequency:

    def __init__(self, frequencies: Dict[str, float], domain: str, average: Dict[str, float] = None) -> None:
        self.domain = domain
        self.frequencies = DomainClassFrequency.normalize_frequency_dict(frequencies, average)

    @staticmethod
    def normalize_frequency_dict(frequency_dict: Dict[str, int], average: Dict[str, float] = None):
        """
        Takes a dictionary where the keys are a string and the values are the occurrence
        of that string (in a text for example) and normalizes the occurrence using the
        total occurrences while substracting the average value
        """
        frequencies = {}

        total = 0
        for key, count in frequency_dict.items():

            normalized_count = count
            if average is not None:
                if key in average:
                    normalized_count = max(0, normalized_count - average[key])
            total += normalized_count
            frequencies[key] = normalized_count

        for key, value in frequencies.items():
            frequencies[key] = value / total

        return frequencies

    def get_most_common(self, top_n: int = 30, exclude_set: int = set(), snomed: SNOMED = None):

        if len(exclude_set) == 0:
            return Counter(self.frequencies).most_common(top_n)

        assert snomed is not None, "If the exclude set is given, the snomed argument must be provided"

        filtered_frequencies = {}
        for id, freq in self.frequencies.items():
            ancestors = snomed.get_ancestors_of_id(id, return_set=True)
            if len(ancestors) == 0:
                continue
            
            ancestors.remove(snomed.base_class.id)
            ancestors.add(id)
            if len(ancestors.intersection(exclude_set)) == 0 and id != snomed.base_class.id:
                filtered_frequencies[id] = freq
        
        return Counter(filtered_frequencies).most_common(top_n)



class DomainClassAnalysis:
    """
    Class containing the `DomainClassFrequency` of all domains 
    """

    def __init__(self, snomed: SNOMED, annotations: Dict[str, List[AnnotatorResult]], normalize_with_average = True) -> None:
        self.snomed = snomed

        concepts_per_domain = self.extract_domain_class_frequencies(annotations)
        self.normalize_with_average = normalize_with_average
        if normalize_with_average:
            self.average_class_frequency = self.compute_average_class_frequency(concepts_per_domain)
        self.domain_class_frequencies: Dict[str, DomainClassFrequency] = self.transform_concepts_per_domain_to_dcf(concepts_per_domain)

    def extract_domain_class_frequencies(self, annotations: Dict[str, List[AnnotatorResult]]) -> Dict[str, List]:
        """
        Retrieves all classes in ontology associated to annotation results including the ancestors of those classes
        for each domain
        """
        concepts_per_domain = defaultdict(list)
        for domain, annotator_results in annotations.items():
            for annotator_result in tqdm(annotator_results):
                snomed_id = annotator_result.snomed_id
                if len(snomed_id) == 0:
                    continue
                    
                concepts_per_domain[domain].append(snomed_id)           
                
                ancestors = self.snomed.get_ancestors_of_id(snomed_id, return_list=True)
                ancestors = ancestors[:min(MAX_ANCESTORS, len(ancestors))]

                if len(ancestors) == 0:
                    continue
                
                for ancestor in ancestors:
                    concepts_per_domain[domain].append(ancestor)
        return concepts_per_domain

    def transform_concepts_per_domain_to_dcf(self, concepts_per_domain: Dict[str, List]):
        """
        Counts the occurrence of each concept in each domain and creates a domain class frequency mapping while 
        substracting the frequency to the average class frequency
        """
        domain_class_frequencies = {}
        for domain, concepts in concepts_per_domain.items():
            average_class_frequency = self.average_class_frequency if self.normalize_with_average else None
            domain_class_frequencies[domain] = DomainClassFrequency(Counter(concepts), domain, average_class_frequency)
        return domain_class_frequencies

    def compute_average_class_frequency(self, concepts_per_domain: Dict[str, List]):
        """
        Computes the average class frequency (used in the normalization setting)

        """
        average_class_frequency = {}
        nb_domains = len(concepts_per_domain.keys())
        for _, concepts in concepts_per_domain.items():
            for concept in concepts:
                if concept in average_class_frequency:
                    average_class_frequency[concept] += 1 / nb_domains
                else:
                    average_class_frequency[concept] = 1 / nb_domains
        return average_class_frequency

    def save(self, path: str):
        return joblib.dump(self.domain_class_frequencies, path)

