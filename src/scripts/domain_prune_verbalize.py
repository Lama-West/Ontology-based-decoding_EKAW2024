import sys  
from pathlib import Path

import joblib
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import pandas as pd

from src.hf_utils import load_hf_checkpoint
from src.ontology.umls import MedCatAnnotator
from src.ontology.snomed import SNOMED
from src.generation import DomainOntologyBasedVerbalizer

from src.prompts import DOMAIN_NORMAL_POST_PROMPT, DOMAIN_NORMAL_PRE_PROMPT, DOMAIN_OURS_POST_PROMPT, DOMAIN_OURS_PRE_PROMPT

parser = argparse.ArgumentParser(description='Applies the pruning and verbalizer for domain adaptation (generates domain adapted summaries)')
parser.add_argument('--dataset', type=str, required=True, help='Path to the csv file of the dataset containing the structured version')
parser.add_argument('--type', type=int, required=True, help='Type of verbalization (0: normal, 1: ext+pru')
parser.add_argument('--out', type=str, required=True, help='Path where the results will be stored')

parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--use_quantization', type=int, required=False, default=0, help='Whether to use quantization or not when loading the model')

parser.add_argument('--dcfs', type=str, required=True, help='Path to domain class frequencies')
parser.add_argument('--snomed', type=str, required=True, help='Path to snomed')
parser.add_argument('--snomed_cache', type=str, required=True, help='Path to snomed cache')
parser.add_argument('--medcat', type=str, required=True, help='Path to medcat')

def verbalize():
    
    args = parser.parse_args()
    print(f'Script called with args : {args}')

    # Load model, ontology and annotator
    model, tokenizer = load_hf_checkpoint(args.model, use_quantization=args.use_quantization == 1, padding_side='left')
    snomed = SNOMED(args.snomed, cache_path=args.snomed_cache, nb_classes=366771)
    annotator = MedCatAnnotator(args.medcat, snomed, device=model.device.type)

    if args.type == 0:
        pre_prompt = DOMAIN_NORMAL_PRE_PROMPT
        post_prompt = DOMAIN_NORMAL_POST_PROMPT
    else:
        pre_prompt = DOMAIN_OURS_PRE_PROMPT
        post_prompt = DOMAIN_OURS_POST_PROMPT
    
    dcfs = joblib.load(args.dcfs)
    verbalizer = DomainOntologyBasedVerbalizer(
        output_path=args.out,
        domain_analysis=dcfs,
        snomed=snomed,
        annotator=annotator,
        model=model,
        tokenizer=tokenizer,
    )

    structured = pd.read_csv(args.dataset)

    verbalizer.start(
        structured=structured,
        pre_prompt=pre_prompt,
        post_prompt=post_prompt,
        normal=args.type == 0,
        text_column='notes',
        summary_column='bhc',
        domain_filter=['Nursing', 'Physician ', 'Radiology', 'ECG']
    )

    
if __name__ == '__main__':
    verbalize()
