import owlready2 as owl
from tqdm import tqdm
import joblib
import os
import json
from collections import defaultdict, deque
from typing import List

class SClass():

    def __init__(self, snomed_class: owl.ThingClass, *args):
        self.owl_class = snomed_class
        self.path = str(snomed_class.iri)
        self.id = self.path[len(SNOMED.BASE_PATH):]
        
        # The label attribute includes the parent class
        self.label_parent = self.get_locstr(snomed_class.label.first())
        self.label = self.get_locstr(snomed_class.prefLabel.first())
        if not self.label:
            self.label = self.label_parent
        
        self.all_labels = self.all_labels = snomed_class.label
        self.alt_label = [self.get_locstr(label) for label in snomed_class.altLabel]
        self.definition = [self.get_locstr(defn) for defn in snomed_class.definition]

    def get_locstr(self, locstr, lang="en"):
        if isinstance(locstr, dict):
            return locstr.get(lang, locstr.get('en', ''))
        return locstr

    def __str__(self) -> str:
        out = f'ID : "{self.id}"\n'
        out += f'Label : {self.label}\n'
        # out += f'Alternative label : [{", ".join(self.alt_label) if self.alt_label is not None and len(self.alt_label) > 0 else ""}]\n'
        if len(self.definition) > 0:
            out += f'Definition : [{", ".join(self.definition) if self.definition is not None and len(self.alt_label) > 0 else ""}]\n'
        
        return out

class SProperty:

    def __init__(self, property_type, ids, labels, o_restriction) -> None:
        self.property_type = property_type
        self.ids_to_ids = ids
        self.labels_to_labels = labels
        self.o_restriction = o_restriction

    def get_value(self):
        if self.property_type == 'or':
            return ' or '.join(self.labels_to_labels.values())
        else:
            return ' '.join(self.labels_to_labels.values())
        
    def __str__(self) -> str:
        out = f'Property type : {self.property_type}\n'
        
        for k, v in self.labels_to_labels.items():
            out += f'\t{k} : {v}\n'

        return out

class SClassEncoder(json.JSONEncoder):
        def default(self, o):
            encoded = o.__dict__
            encoded['owl_class'] = ''
            return encoded


class SPropertyEncoder(json.JSONEncoder):
        def default(self, o):
            encoded = o.__dict__
            encoded['o_restriction'] = ''
            return encoded

class SNOMED:

    BASE_CLASS_ID = '138875005'
    BASE_PATH = 'http://snomed.info/id/'

    ID_TO_CLASS_PATH = 'id_to_classes.snomed'
    PARENTS_OF_PATH = 'parents_of.snomed'
    CHILDREN_OF_PATH = 'children_of.snomed'
    PROPERTY_OF_PATH = 'properties_of.snomed'

    def __init__(self, path: str, cache_path = './', rebuild=False, nb_classes: int = -1):
        self.ontology = owl.get_ontology(path).load()

        self.base_class = self.get_class_from_id(SNOMED.BASE_CLASS_ID, refetch=True)
        self.nb_classes = nb_classes if nb_classes > -1 else len(list(self.ontology.classes()))
        
        self.cache_path = cache_path
        self.id_to_classes_path = self.cache_path + SNOMED.ID_TO_CLASS_PATH
        self.parents_of_path = self.cache_path + SNOMED.PARENTS_OF_PATH
        self.children_of_path = self.cache_path + SNOMED.CHILDREN_OF_PATH  
        self.properties_of_path = self.cache_path + SNOMED.PROPERTY_OF_PATH

        self.id_to_classes = {}
        self.parents_of = defaultdict(list)
        self.children_of = defaultdict(list)
        self.properties_of = defaultdict(list)
        self.not_found = {}
        
        
        # with self.ontology:
            # owl.sync_reasoner()

        self.build(rebuild=rebuild)

    def summary(self):
        print(f'Number of classes : {len(self.id_to_classes)}')
        print(f'Number of parents relationships : {len(self.parents_of)}')
        print(f'Number of children relationships : {len(self.children_of)}')
        print(f'Number of properties : {len(self.properties_of)}')

    def get_subjects(self, concept):
        if type(concept) is owl.class_construct.Restriction:
            return [concept.value]
        elif (type(concept) is owl.class_construct.Or) or (type(concept) is owl.class_construct.And):
            return concept.get_Classes()
        else:
            return [concept]
        
    def build(self, rebuild=False):

        if self.verify_cache() and not rebuild:
            self.load()
        else:
            for o_class in tqdm(self.ontology.classes(), total=self.nb_classes):
                sclass = SClass(o_class)
                self.id_to_classes[sclass.id] = sclass
            
            for o_class in tqdm(self.ontology.classes(), total=self.nb_classes):
                sclass = SClass(o_class)
                for parent_classes in (self.ontology.get_parents_of(o_class) + o_class.equivalent_to):
                    
                    for parent_class in self.get_subjects(parent_classes):
                        if isinstance(parent_class, owl.Restriction):
                            parent_value = parent_class.value
                            text_property = ''
                            if isinstance(parent_value, owl.And) or isinstance(parent_value, owl.Or):
                                property_type = 'and'
                                # Remove properties that are not restrictions or object properties
                                classes = list(filter(lambda x: isinstance(x, owl.Restriction)\
                                                    and isinstance(x.property, owl.ObjectPropertyClass), \
                                        parent_value.get_Classes()))
                                
                                # Generate labels dictionary 
                                labels = {self.get_class_from_id(x.property._name, refetch=True).label: \
                                          self.get_class_from_id(x.value._name, refetch=True).label for x in classes}
                                
                                # Generate ids dictionary
                                ids = {x.property._name: x.value._name for x in classes}
                            elif isinstance(parent_value, owl.Restriction):
                                property_type = 'res_simple'
                                labels = {self.get_class_from_id(parent_value.property._name, refetch=True).label: \
                                    self.get_class_from_id(parent_value.value._name, refetch=True).label}
                                ids = {parent_value.property._name: parent_value.value._name}
                            elif isinstance(parent_value, owl.ThingClass):
                                property_type = 'simple'
                                text_property = self.get_class_from_id(parent_value._name, refetch=True).label
                                ids = {parent_value._name: parent_value._name}
                                labels = {text_property: text_property}
                            else:
                                p = self.get_class_from_id(parent_class.property._name, refetch=True)
                                self.not_found[p.id] = (p, parent_value)
                                continue

                            classes = []
                            classes.extend(list(ids.keys()))
                            classes.extend(list(ids.values()))

                            for c in classes:
                                extracted_sclass = self.get_class_from_id(c, refetch=True)
                                if extracted_sclass is None:
                                    print(f'{c} is none')
                                self.id_to_classes[c] = extracted_sclass
                            
                            property = SProperty(property_type=property_type, 
                                                 ids=ids, 
                                                 labels=labels,
                                                 o_restriction=parent_class)
                            self.properties_of[sclass.id].append(property)
                        else:
                            parent_sclass = SClass(parent_class)
                            self.parents_of[sclass.id].append(parent_sclass.id)
                            self.children_of[parent_sclass.id].append(sclass.id)

    def verify_cache(self):
        return os.path.exists(self.id_to_classes_path) and \
            os.path.exists(self.parents_of_path) and \
            os.path.exists(self.children_of_path) and \
            os.path.exists(self.properties_of_path)
    
    def load(self):
        with open(self.id_to_classes_path, 'r') as f:
            tmp = json.load(f)
            for k, v in tmp.items():
                if v is None:
                    print('v is none : ', k)
                obj = object.__new__(SClass)
                obj.__dict__ = v
                self.id_to_classes[k] = obj
            
        with open(self.parents_of_path, 'r') as f:
            self.parents_of = json.load(f)
            
        with open(self.children_of_path, 'r') as f:
            self.children_of = json.load(f)
        
        with open(self.properties_of_path, 'r') as f:
            tmp = json.load(f)
            for k, v in tmp.items():
                for property in v:
                    obj = object.__new__(SProperty)
                    obj.__dict__ = property
                    self.properties_of[k].append(obj)

    def save(self):
        with open(self.id_to_classes_path, 'w+') as f:
            f.write(json.dumps(self.id_to_classes, indent=4, cls=SClassEncoder))

        with open(self.parents_of_path, 'w+') as f:
            f.write(json.dumps(self.parents_of, indent=4, cls=SClassEncoder))

        with open(self.children_of_path, 'w+') as f:
            f.write(json.dumps(self.children_of, indent=4, cls=SClassEncoder))

        with open(self.properties_of_path, 'w+') as f:
            f.write(json.dumps(self.properties_of, indent=4, cls=SPropertyEncoder))

    def convert_ids_to_classes(self, ids, refetch=False):
        classes = []
        for id in ids:
            classes.append(self.get_class_from_id(id, refetch=refetch))
        return classes
    
    def get_class_from_id(self, id, refetch=False) -> SClass:
        """
        Search the ontology by id (iri). The prefix "http://snomed.info/id/" must be omitted
        """
        if refetch:
            return SClass(self.__get_class_from_id(id))
        else:
            if id not in self.id_to_classes:
                return SClass(self.__get_class_from_id(id))
            return self.id_to_classes[id]

    def get_properties_of_id(self, id) -> List[SProperty]:
        # ancestors = self.get_ancestors_of_id(id)
        # print(ancestors)
        if id not in self.properties_of:
            # print(f'id {id} not found')
            return []
        else:
            properties = self.properties_of[id]
            if isinstance(properties, list):
                return properties
            return [properties]

    def get_parents_of_id(self, id: str):
        if id not in self.parents_of:
            return self.__get_class_from_id(id).parents_of
        parent_ids = self.parents_of[id]
        return self.convert_ids_to_classes(parent_ids)

    def get_ancestors_of_id(self, id: str, return_set=False, return_list=False):
        if return_set:
            return set(self.__get_all_ancestors_of_id_as_list(id))
        
        if return_list:
            return self.__get_all_ancestors_of_id_as_list(id)
        
        if id not in self.parents_of:
            return self.__get_class_from_id(id).parents_of

        ancestors = []

        queue = deque([(id, ancestors)])

        while queue:
            current_id, current_list = queue.popleft()
            if current_id in self.parents_of:
                parent_ids = self.parents_of[current_id]
                for parent_id in parent_ids:
                    parent_list = []
                    current_list.append({parent_id: parent_list})
                    queue.append((parent_id, parent_list))

        return ancestors
   
        
    def __get_class_from_id(self, id):
        return self.ontology.search(iri=SNOMED.BASE_PATH + id).first()

    def __get_all_ancestors_of_id_as_list(self, id: str):
        ancestors = list()
        queue = deque([id])

        while queue:
            current_id = queue.popleft()
            if current_id in self.parents_of:
                parent_ids = self.parents_of[current_id]
                for parent_id in parent_ids:
                    if parent_id not in ancestors:
                        ancestors.append(parent_id)
                        queue.append(parent_id)

        return ancestors

    def get_children_of_id(self, id: str, ids_only=False):
        if id not in self.children_of:
            return []
        child_ids = self.children_of[id]
        if ids_only:
            return child_ids
        return self.convert_ids_to_classes(child_ids)
