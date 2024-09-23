import sys  
from pathlib import Path

import joblib
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import pandas as pd

from src.hf_utils import load_hf_checkpoint
from src.ontology.umls import MedCatAnnotator
from src.ontology.snomed import SNOMED
from src.generation import BHCOntologyBasedVerbalizer
from src.prompts import BHC_EXT_PRE_PROMPT, BHC_EXT_POST_PROMPT, BHC_EXT_PRU_POST_PROMPT, BHC_EXT_PRU_PRE_PROMPT, BHC_NORMAL_POST_PROMPT, BHC_NORMAL_PRE_PROMPT

parser = argparse.ArgumentParser(description='Applies the pruning and verbalizer on a set of extracted attributes')
parser.add_argument('--dataset', type=str, required=True, help='Path to the csv file of the dataset containing the structured version')
parser.add_argument('--type', type=int, required=True, help='Type of verbalization (0: normal, 1: ext, 2: ext+pru)')
parser.add_argument('--out', type=str, required=True, help='Path where the results will be stored')

parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--use_quantization', type=int, required=False, default=0, help='Whether to use quantization or not when loading the model')

parser.add_argument('--bhc', type=str, required=True, help='Path to BHC frequencies')
parser.add_argument('--snomed', type=str, required=True, help='Path to snomed')
parser.add_argument('--snomed_cache', type=str, required=True, help='Path to snomed cache')
parser.add_argument('--medcat', type=str, required=True, help='Path to medcat')

parser.add_argument('--top_n', type=int, required=False, default=30, help='Top n concepts (based on frequency) to keep when initially annotating the note')

ALL_METHODS = ['normal', 'normal_beam', 'constrained_beam']

def verbalize():
    
    args = parser.parse_args()
    print(f'Script called with args : {args}')
    extract = args.type > 0
    prune = args.type > 1
    merged = args.type > 2
    print(f'Extract = {extract}, Prune = {prune}, Merged = {merged}')

    # Load model, ontology and annotator
    model, tokenizer = load_hf_checkpoint(args.model, use_quantization=args.use_quantization == 1, padding_side='left')
    snomed = SNOMED(args.snomed, cache_path=args.snomed_cache, nb_classes=366771)
    annotator = MedCatAnnotator(args.medcat, snomed, device=model.device.type)

    if args.type == 0:
        pre_prompt = BHC_NORMAL_PRE_PROMPT
        post_prompt = BHC_NORMAL_POST_PROMPT
    elif args.type == 1:
        pre_prompt = BHC_EXT_PRE_PROMPT
        post_prompt = BHC_EXT_POST_PROMPT
    else:
        pre_prompt = BHC_EXT_PRU_PRE_PROMPT
        post_prompt = BHC_EXT_POST_PROMPT

    verbalizer = BHCOntologyBasedVerbalizer(
        output_path=args.out,
        snomed=snomed,
        annotator=annotator,
        model=model,
        tokenizer=tokenizer
    )

    structured = pd.read_csv(args.dataset)

    bhc_dcf = joblib.load(args.bhc)

    verbalizer.start(
        structured=structured,
        pre_prompt=pre_prompt,
        post_prompt=post_prompt,
        extract=extract,
        prune=prune,
        merged=merged,
        domain_frequencies=bhc_dcf,
        text_column='notes',
        summary_column='bhc',
        top_n_concepts=args.top_n,
        ancestor_level=2,
    )

    
if __name__ == '__main__':
    verbalize()
