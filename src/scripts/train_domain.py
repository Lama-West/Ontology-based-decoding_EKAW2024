import sys  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import os
import joblib
import pandas as pd
import os.path

from src.domain_evaluation_model import DomainEvaluationDataset, DomainEvaluationModel
from src.preprocessor import remove_deidentification_headers, lower_all, Processor

import pandas as pd

parser = argparse.ArgumentParser(description='Train domain evaluator model')
parser.add_argument('--train', type=str, required=True, help='Path to train file')
parser.add_argument('--eval', type=str, required=True, help='Path to validation file')
parser.add_argument('--out', type=str, required=True, help='Path where the trained mdoel will be saved')
parser.add_argument('--cache', type=str, required=True, help='Path to the cache folder (for the dataset library)')
parser.add_argument('--model', type=str, required=True, help='Path to huggingface model used for training')
parser.add_argument('--batch_size', type=int, required=True, default=128, help='Batch size to use during training')

def train():

    args = parser.parse_args()
    print(f'Script called with args : {args}')
    domains = ['Nursing', 'Physician ', 'Radiology', 'ECG']
    
    dataset = DomainEvaluationDataset(
        train_file_path=args.train,
        eval_file_path=args.eval,
        text_column='TEXT',
        domain_column='CATEGORY',
        domain_filter=domains,
        filter=False,
        cache_dir=args.cache
    )


    processor = Processor([remove_deidentification_headers, lower_all])
    dataset.preprocess_text(processor)

    domain_evaluation_model = DomainEvaluationModel(
        backbone_model_path=args.model,
        domains=domains,
    )

    trainer = domain_evaluation_model.train(dataset, out_dir=args.out, batch_size=args.batch_size)
    trainer.save_model(args.out)

if __name__ == '__main__':
    train()
