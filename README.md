# Ontology-based constrained decoding - EKAW 2024

## Overview
This is the repository for the paper "Ontology-Constrained Generation of Domain-Specific Clinical Summaries" accepted at EKAW 2024. This method is a special domain adaptation technique based on dyanimc prompting and constrained beam search. The beam search algorithm is guided with an ontology.

LAMA-WeSt lab: http://www.labowest.ca/

## Pre-requisites

You will need access to the following ressources :
- [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
- [SNOMED-CT](https://www.snomed.org/)
  - You must have access to the .owx file. When downloading the ontology, it will not be given in an .owx format. Use a converter (e.g. protege) to get the .owx format.
- [MedCAT Annotator Tool](https://github.com/CogStack/MedCAT)

## Getting started
Clone the repository
```
git clone https://github.com/Lama-West/Ontology-based-decoding_EKAW2024.git
cd Ontology-based-decoding_EKAW2024
```
Install requirements
```
pip install -r requirements.txt
```

## Prompt formats
All prompt formats are given in the file `src/prompts.py`. 

### Extraction
For example, the prompt used to extract all procedures from a clinical note is given by :
```
"""Here is a clinical note about a patient : 
-------------------
{clinical_note}
-------------------
In a short sentence, summarize everything related to the "Procedure" concept mentioned the clinical note. \n\n If the concept is not mentioned in the note, respond with 'N/A'.
"""

```

## Inference
### Domain Class Frequency
To adapt a summary to a domain, we need to find the most frequent concepts for a certain domain. This analysis is performed by computing the frequencies of concepts in multiple texts from the same domain. On the MIMIC-III dataset, these are the most frequent concepts for each domain : 
![Most frequent concepts in MIMIC-III per category](./resources/domain_adaptation_analysis_mimic.svg)


```python
# Load ontology
snomed_path = config.get('paths', 'snomed_ct')
snomed = SNOMED(snomed_path, cache_path='tmp/', nb_classes=366771)
```
```python
# Load annotator
annotator = MedCatAnnotator(
    medcat_path, 
    snomed,
    meta_cat_config_dict={
        'general': {
            'device': model.device.type
        }
    }
)
```
```python
# Group texts by domain (CATEGORY column in MIMIC-III).
category_attributes = df.groupby('CATEGORY')['TEXT'].apply(
    lambda texts: [attr for text in texts for attr in annotator.annotate(text)]
).to_dict()

analysis = DomainClassAnalysis(snomed, category_attributes, normalize_with_average=True)

print(Counter(analysis.domain_class_frequencies['Nursing'].frequencies).most_common(5))
# 5 most frequent concepts of Nursing category
```

### Extraction (constrained decoding)
Assuming that these variables are set :
- `model_path` : Huggingface checkpoint (local)
- `snomed_path` : Path to .owx of snomed ontology
- `medcat_path` : Path to medcat annotator
```python
# Load model
model, tokenizer = load_hf_checkpoint(model_path, padding_side='left', use_quantization=True, device_map={"": 0})
```
```python
# Load ontology
snomed_path = config.get('paths', 'snomed_ct')
snomed = SNOMED(snomed_path, cache_path='tmp/', nb_classes=366771)
```
```python
# Load annotator
annotator = MedCatAnnotator(
    medcat_path, 
    snomed,
    meta_cat_config_dict={
        'general': {
            'device': model.device.type
        }
    }
)
```
```python
# Configuration for inference
config = DiverseBeamSearchConfig()
config.use_scorer = True

constrained_model = OntologyConstrainedModel(
    model, 
    tokenizer, 
    snomed=snomed, 
    annotator=annotator, 
)

prompter = OntologyBasedPrompter(
    constrained_model,
    snomed_ct=snomed,
    annotator=annotator, 
)

prompter.start_multiple(
    clinical_notes, # List of clinical notes 
    batch_size=..., # Number of concepts to treat in parallel
    top_n=..., # Number of concepts to extract per note
    generation_config=config,
)

print(prompter.attributes) 
# [ 
#   {'Procedure': ..., 'Clinical finding': ...} # Note 1
#   {'Observable entity': ...} # Note 2
#   ...
# ]
```


### Pruning and Verbalizer
Assuming that the attributes were extracted (in a DataFrame) and that the domain adaptation analysis was performed.
```python
dcfs = ... # Domain class frequencies
verbalizer = DomainOntologyBasedVerbalizer(
    output_path=..., # Output path
    domain_analysis=dcfs,
    snomed=snomed,
    annotator=annotator,
    model=model,
    tokenizer=tokenizer,
)

verbalizer.start(
    structured=results, # DataFrame containig the results
    pre_prompt=DOMAIN_PRE_PROMPT,
    post_prompt=DOMAIN_POST_PROMPT,
    text_column='notes',
    summary_column='bhc', # If for BHC
    domain_filter=['Nursing', 'Physician ', 'Radiology', 'ECG']
)

```

##
