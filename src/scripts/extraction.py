import sys  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
import argparse
import os
import pandas as pd
import os.path

from src.medical_beam_scorer import DiverseBeamSearchConfig
from src.hf_utils import load_hf_checkpoint
from src.preprocessor import preprocess_clinical_notes
from src.ontology.umls import MedCatAnnotator
from src.ontology.snomed import SNOMED
from src.generation import OntologyBasedPrompter, OntologyConstrainedModel

parser = argparse.ArgumentParser(description='Extraction')
parser.add_argument('--dataset', type=str, required=True, help='Path to the csv file of the dataset')
parser.add_argument('--method', type=str, required=True, help='Which method to use for inference (normal, normal_beam, constrained_beam)')
parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--snomed', type=str, required=True, help='Path to snomed')
parser.add_argument('--snomed_cache', type=str, required=True, help='Path to snomed cache')
parser.add_argument('--medcat', type=str, required=True, help='Path to medcat')
parser.add_argument('--out', type=str, required=True, help='Path where the results will be stored')
parser.add_argument('--max_rows', type=int, required=False, default=10000, help='Maximal number of rows that will be processed in the dataset')
parser.add_argument('--starting_row', type=int, required=False, default=0, help='Which row will be the starting processing row in the dataset')
parser.add_argument('--nb_beams', type=int, required=False, default=10, help='Number of beams during generation (only applied if method is not normal)')
parser.add_argument('--nb_beam_groups', type=int, required=False, default=2, help='Number of beam groups during generation (only applied if method is not normal)')
parser.add_argument('--top_n', type=int, required=False, default=30, help='Top n concepts (based on frequency) to keep when initially annotating the note')
parser.add_argument('--batch_size', type=int, required=False, default=2, help='Number of questions to ask in parallel ')
parser.add_argument('--use_quantization', type=bool, required=False, default=False, help='Whether to use quantization or not when loading the model')
parser.add_argument('--hbf', type=int, required=False, default=1)
parser.add_argument('--pbf', type=int, required=False, default=1)
parser.add_argument('--gbf', type=int, required=False, default=1)

ALL_METHODS = ['normal', 'normal_beam', 'constrained_beam']

def generate():
    
    args = parser.parse_args()
    print(f'Script called with args : {args}')

    # Load model, ontology and annotator
    model, tokenizer = load_hf_checkpoint(args.model, use_quantization=args.use_quantization, padding_side='left')
    snomed = SNOMED(args.snomed, cache_path=args.snomed_cache, nb_classes=366771)
    annotator = MedCatAnnotator(args.medcat, snomed, device=model.device.type)

    # Load data
    df = pd.read_csv(args.dataset)
    df = df[df.clinical_notes.notna()]
    df = df[df.summary_bhc.notna()]
    print('Number of elements in dataset : ', len(df))

    # Load output file
    if os.path.exists(args.out):
        df_results = pd.read_csv(args.out)
    else:
        df_results = pd.DataFrame([], columns=['id', 'notes', 'bhc', 'attributes', 'attributes_by_id'])
    
    starting_row = args.starting_row + len(df_results)
    print('Starting at row : ', starting_row)
    
    TOP_N = args.top_n
    BATCH_SIZE = args.batch_size
    
    method = args.method
    if method not in ALL_METHODS:
        print(f'Method not valid ({method}). Expected one of these methods : ', ALL_METHODS)
        return

    scorer_config = DiverseBeamSearchConfig()
    scorer_config.score_boost_factors = [args.hbf, args.pbf, args.gbf]
    scorer_config.use_scorer = True

    normal_beam_config = DiverseBeamSearchConfig()
    normal_beam_config.use_scorer = True
    normal_beam_config.normal_beam_search = True

    for i in range(starting_row, args.max_rows):
        start = time.time()
        id = df.hadm_id.iloc[i]
        notes = df.clinical_notes.iloc[i]
        bhc = df.summary_bhc.iloc[i]
        individual_notes = preprocess_clinical_notes(notes, tokenizer)

        if len(individual_notes) == 0:
            continue

        aces_model = OntologyConstrainedModel(
            model, 
            tokenizer, 
            snomed=snomed, 
            annotator=annotator, 
        )

        prompter = OntologyBasedPrompter(
            aces_model,
            snomed_ct=snomed,
            annotator=annotator, 
        )
        
        if method == 'normal':
            # No beam search, normal generation
            prompter.start_multiple(
                individual_notes, 
                batch_size=BATCH_SIZE, 
                top_n=TOP_N,
            )
        elif method == 'normal_beam':
            # Normal beam search
            prompter.start_multiple(
                individual_notes, 
                batch_size=BATCH_SIZE, 
                top_n=TOP_N, 
                generation_config=normal_beam_config,
            )
        else:
            # Constrained beam search
            prompter.start_multiple(
                individual_notes, 
                batch_size=BATCH_SIZE, 
                top_n=TOP_N, 
                generation_config=scorer_config,
            )

        new_row = {
            'id': id,
            'notes': notes, 
            'bhc': bhc, 
            'attributes': prompter.attributes,
            'attributes_by_id': prompter.attributes_by_id,
        }

        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
        end = time.time()
        print('Time to generate : ', end-start)
        start = time.time()
        df_results.to_csv(args.out)
        end = time.time()
        print('Time to save : ', end-start)

    
if __name__ == '__main__':
    generate()
