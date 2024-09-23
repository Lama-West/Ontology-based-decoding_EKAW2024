# Python
from typing import List

# NLP
import torch.utils
import torch.utils.data
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, load_dataset, concatenate_datasets

# ML
import torch
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Others
from tqdm import tqdm

# Source
from src.ontology.umls import Annotator
from src.utils import PandasToTorchDataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(predictions, labels)
    }

TRAIN_SPLIT_NAME = 'train'
EVAL_SPLIT_NAME = 'eval'


class DomainEvaluationPreProcessor:

    """
    Will pass domain-tailored texts through a paraphraser to remove 
    format-based domain detection (models trained to detect the domain
    of certain texts tend to base their prediction based on the format
    of the text, not on the actual concepts present in the texts)
    """

    def __init__(
        self,
        data_path: str,
        paraphraser_model_path: str,
        text_column: str,
        domain_column: str,
        domain_filter: List[str],
        prompt: str = None
    ):
        self.text_column = text_column
        self.domain_column = domain_column
        self.domain_filter = domain_filter
        self.data_path = data_path

        if prompt is None:
            self.prompt = """"""
        else:
            self.prompt = prompt

        # Load data
        self.data = pd.read_csv(data_path)
        self.data = self.filter_data(self.data, self.domain_column, self.domain_filter)

        # Load paraphraser
        self.load_model(paraphraser_model_path)

    def load_model(self, path):
        self.summarizer = pipeline(
            "summarization", 
            model=path, 
            # local_files_only=True,
            device_map='auto'
        )

        # self.model, self.tokenizer = load_hf_checkpoint(path, use_quantization=True, padding_side='left')

    def filter_data(self, data, column, filter_values):
        """
        Removes every text in `self.data` that is not linked to a domain
        in `self.domain_filter`
        """
        return data[data[column].isin(filter_values)]

    def start(self, batch_size: int = 1, out_path: str = 'domain_evaluation_dataset.csv'):

        def preprend_prompt(x):
            return [s[0] for s in x], [self.prompt + s[1] for s in x]

        torch_dataset = PandasToTorchDataset(self.data, [self.domain_column, self.text_column])
        data_loader = DataLoader(
            torch_dataset,
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=preprend_prompt
        )

        df = pd.DataFrame([], columns=[self.text_column, self.domain_column, 'old_text'])

        for data in tqdm(data_loader):
            domains_of_texts, texts = data
            paraphrased_texts = list(map(lambda x: x['summary_text'], self.summarizer(
                texts, 
                max_length=1024,
                min_length=512, 
                truncation=True, 
                batch_size=batch_size
            ))) 
            new_rows = pd.DataFrame({self.text_column: paraphrased_texts, self.domain_column: domains_of_texts, 'old_texts': texts})
            df = pd.concat([df, new_rows], ignore_index=True)
            df.to_csv(out_path)

    def save_data(self, path: str):
        self.data.to_csv(path)

class DomainEvaluationDataset:
    """
    Loads a csv file and prepares the dataset for training by removing columns that are not needed during training,
    only keeping the domains needed during training and keeping at most a certain number of samples per domain
    """

    def __init__(
        self, 
        train_file_path: str,
        eval_file_path: str,
        text_column: str,
        domain_column: str,
        domain_filter: List[str],
        cache_dir: str,
        filter=True,
    ):
        """
        domain_filter   :   Domains to keep during training
        """

        data_files = {
            TRAIN_SPLIT_NAME: train_file_path, 
            EVAL_SPLIT_NAME: eval_file_path, 
        }

        self.domain_filter = domain_filter
        self.text_column = text_column
        self.domain_column = domain_column
        
        self.internal_dataset: Dataset = load_dataset('csv', data_files=data_files, cache_dir=cache_dir)
        print('Loaded dataset')

        self.remove_useless_columns()
        if filter:
            self.filter()

    def remove_useless_columns(self):
        columns = self.internal_dataset.column_names[TRAIN_SPLIT_NAME]
        columns.remove(self.text_column)
        columns.remove(self.domain_column)
        self.internal_dataset[TRAIN_SPLIT_NAME] = self.internal_dataset[TRAIN_SPLIT_NAME].remove_columns(columns)
        self.internal_dataset[EVAL_SPLIT_NAME] = self.internal_dataset[EVAL_SPLIT_NAME].remove_columns(columns)

    def filter_and_limit(self, dataset, domain, max_samples=10000):
        """
        Selects all rows from a certain domain and selects at most `max_samples`
        """
        domain_dataset = dataset.filter(lambda x: x[self.domain_column] == domain)
        return domain_dataset.shuffle().select(range(min(len(dataset), max_samples)))

    def filter(self):
        """
        Filters the rows of the dataset to only contain the domains mentioned in `self.domain_filter`
        """
        filtered_datasets = [self.filter_and_limit(self.internal_dataset[TRAIN_SPLIT_NAME], domain, max_samples=100000) for domain in self.domain_filter]
        self.internal_dataset[TRAIN_SPLIT_NAME] = concatenate_datasets(filtered_datasets)

        filtered_datasets = [self.filter_and_limit(self.internal_dataset[EVAL_SPLIT_NAME], domain, max_samples=10000) for domain in self.domain_filter]
        self.internal_dataset[EVAL_SPLIT_NAME] = concatenate_datasets(filtered_datasets)

    def shuffle(self):
        self.internal_dataset[TRAIN_SPLIT_NAME] = self.internal_dataset[TRAIN_SPLIT_NAME].shuffle()
        self.internal_dataset[EVAL_SPLIT_NAME] = self.internal_dataset[EVAL_SPLIT_NAME].shuffle()

    def preprocess_text(self, func):
        def apply_func(sample):
            sample[self.text_column] = func(sample[self.text_column])
            return sample

        return self.internal_dataset.map(apply_func, batched=True)

    def __get_item__(self, column: str):
        return self.internal_dataset[column]

class DomainEvaluationModel:
    """
    Wrapper class to train a text-to-domain prediction model or do inference
    """
    
    def __init__(
        self, 
        backbone_model_path: str, 
        domains: List[str],
        # annotator: Annotator,
        max_length: int = 512
    ) -> None:

        self.backbone_model_path = backbone_model_path
        self.domains = domains
        self.nb_domains = len(self.domains)
        # self.annotator = annotator
        self.max_length = max_length
        self.dataset = None
        self.processed = False

        self.load_backbone_model()

    def load_backbone_model(self):
        self.label2id = {k: v for k, v in zip(self.domains, range(self.nb_domains))}
        self.id2label = {v: k for k, v in zip(self.domains, range(self.nb_domains))}

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_model_path)

        self.backbone_model = AutoModelForSequenceClassification.from_pretrained(
            self.backbone_model_path,
            num_labels=self.nb_domains, 
            id2label=self.id2label, 
            label2id=self.label2id,
        )
        self.backbone_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def chunk_examples(self, samples, domains):
        tokenized = self.tokenizer(
            samples, 
            truncation=True, 
            max_length=512,
            return_overflowing_tokens=True,
        )
        labels = []
        for i in range(len(tokenized['input_ids'])):
            overflow_to_sample_mapping = int(tokenized['overflow_to_sample_mapping'][i])
            label = domains[overflow_to_sample_mapping]
            labels.append(self.label2id[label])
        return {
            'input_ids': tokenized['input_ids'], 
            'attention_mask': tokenized['attention_mask'], 
            'label': labels
        }
    
    def prepare(self, dataset: DomainEvaluationDataset):
        if self.dataset is None:
            self.processed = False
            self.dataset = dataset
        
        self.dataset.shuffle()

        self.dataset.internal_dataset = self.dataset.internal_dataset.map(
            lambda x: self.chunk_examples(x[self.dataset.text_column], x[self.dataset.domain_column]), 
            batched=True, 
            remove_columns=[self.dataset.text_column, self.dataset.domain_column]
        )

        self.processed = True

    def predict(self, text: str, use_concepts_only=False, annotator: Annotator = None):
        """
        Predicts the domain of a text by splitting all texts in sequences of 512 tokens
        and returns the mean logits
        """
        if use_concepts_only:
            assert annotator is not None, "When predicting using concepts only, the annotator is needed"
            annotations = annotator.annotate(text)
            input_text = ', '.join(map(lambda x: x.concept, annotations))
        else:
            input_text = text

        model_input = self.tokenizer(
            input_text, 
            truncation=True, 
            padding=True,
            max_length=512,
            return_tensors='pt',
            return_overflowing_tokens=True,
        )

        input_ids = model_input['input_ids'].to(self.backbone_model.device)
        attention_mask = model_input['attention_mask'].to(self.backbone_model.device)

        with torch.no_grad():
            mean_logits = self.backbone_model(input_ids, attention_mask).logits.mean(dim=0)
            return mean_logits

    def train(self, dataset: DomainEvaluationDataset, out_dir: str = 'domain_evaluation_model', batch_size=128):
        """
        Trains the model on a dataset

        Args
            - dataset       : Dataset used to train
            - out_dir       : Directory where checkpoints will be stored
            - batch_size    : Batch size to use during training
        """
        if not self.processed:
            self.prepare(dataset)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=1,
            evaluation_strategy="steps",
            eval_steps=100,
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="steps",
            save_steps=500,
            report_to="none"
        )

        trainer = Trainer(
            model=self.backbone_model,
            args=training_args,
            train_dataset=self.dataset.internal_dataset[TRAIN_SPLIT_NAME],
            eval_dataset=self.dataset.internal_dataset[EVAL_SPLIT_NAME],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        return trainer
