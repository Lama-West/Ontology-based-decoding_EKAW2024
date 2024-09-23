import sys  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse

from src.domain_evaluation_model import DomainEvaluationPreProcessor


parser = argparse.ArgumentParser(description='Paraphrases clinical notes using a paraphraser')
parser.add_argument('--dataset', type=str, required=True, help='Path to the mimic notes')
parser.add_argument('--model', type=str, required=True, help='Path to the paraphraser\'s checkpoint')
parser.add_argument('--out', type=str, required=True, help='Path where the results will be stored')
parser.add_argument('--batch_size', type=int, required=True, default=16, help='Batch size used')

def generate_dataset():
    args = parser.parse_args()

    data_path = args.dataset
    model_path = args.model
    preprocessor = DomainEvaluationPreProcessor(
        data_path, 
        paraphraser_model_path=model_path,
        domain_column='CATEGORY',
        text_column='TEXT',
        domain_filter=['Nursing', 'Physician ', 'Radiology', 'ECG']
    )

    preprocessor.start(batch_size=args.batch_size, out_path=args.out)

if __name__ == '__main__':
    generate_dataset()
