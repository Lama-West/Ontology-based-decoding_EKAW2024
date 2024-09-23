# General
import gc


# Libs
import torch

# Huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from trl import SFTTrainer


# LoRA
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()


def load_hf_checkpoint(
    checkpoint: str, 
    local: bool = True, 
    pad_equal_eos: bool = True, 
    use_quantization: bool = False, 
    quantization_config = None,
    load_bfloat16: bool = True,
    device_map = 'auto',
    padding_side = 'right'
):
    """
    Loads the model and tokenizer a huggingface checkpoint

    Args
        - checkpoint            : Huggingface checkpoint 
        - local                 : Whether the checkpoint is installed locally or not
        - pad_equal_eos         : Whether the padding token should equal the end of sequence token
        - use_quantization      : Whether to use quantization or not (if `quantization_config` is not provided, a default will be used)
        - quantization_config   : Quantization config to use when `use_quantization` is True
        - load_bfloat16         : Whether to load model using bfloat16
        - device_map            : The device map to load the model
        - padding_side          : The padding side of the tokenizer

    Returns
    Tuple containing the model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, 
        local_files_only=local,
        padding_side=padding_side,
    )
    
    config = None
    if use_quantization:
        if quantization_config:
            config = quantization_config
        else:
            compute_dtype = getattr(torch, "float16")
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, 
        local_files_only=local, 
        trust_remote_code=True,
        quantization_config=config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if load_bfloat16 else torch.float16,
    )

    if pad_equal_eos:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
    
def setup_model_for_qlora_training(model, lora_config: LoraConfig = None, return_lora_config=False):
    """
    Prepares a model for peft training using qlora
    
    Args
        - lora_config           : LoRA config to use
        - return_lora_config    : Whether to return the lora config or not
    """
    kbit_model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=8,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    if lora_config:
        config = lora_config
    
    if return_lora_config:
        return get_peft_model(kbit_model, config), config
    
    return get_peft_model(kbit_model, config)
    

def ask_model_batch(model, tokenizer, x: list[str], max_new_tokens=1536, max_length=None):
    """
    Inference pass for a set of inputs. Returns only the newly generated tokens
    """
    clear_gpu_cache()
    
    # 
    model_input = tokenizer(x, padding=True, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            **model_input, 
            max_new_tokens=max_new_tokens,
        )
        # Only retrieve the newly generated tokens
        new_tokens = generated[:, model_input['input_ids'].shape[-1]:]

        results = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        del model_input
        return results



class TokenizerLoader:
    """
    Loads a local huggingface tokenizer 
    """

    def __init__(self, tokenizer_checkpoint: str, pad_equal_eos = True):
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, local_files_only=True)
        if pad_equal_eos:
            self.tokenizer.pad_token = self.tokenizer.eos_token

class TestingModelLoader:
    """
    Loads a peft model for inference
    """
    def __init__(
        self,
        model_checkpoint: str,
        peft_model_checkpoint: str = None,
        quantization_config: BitsAndBytesConfig = None, 
        use_default_quantization_config: bool = True,
        use_peft: str = False,
        local_files_only: bool = True,
    ):
        """
        Args
            - model_checkpoint                  : Huggingface model checkpoint
            - peft_model_checkpoint             : Peft model checkpoint
            - quantization_config               : Quantization config to use
            - use_default_quantization_config   : Whether to use the default quantization config or not
            - use_peft                          : Whether to load the model with peft or not
            - local_files_only                  : Whether the model is local or not
        
        """
        self.model_checkpoint = model_checkpoint
        self.peft_model_checkpoint = peft_model_checkpoint
        self.use_peft = use_peft
        
        if quantization_config is None:
            if use_default_quantization_config:
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
        else:
            self.quantization_config = quantization_config

        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            local_files_only=local_files_only,
            quantization_config=quantization_config,
        )
        
        if use_peft:
            self.model = PeftModel.from_pretrained(self.model, peft_model_checkpoint)

        
class TrainingModelLoader:
    """
    Loads a model for training
    """

    DEFAULT_LORA_CONFIG = LoraConfig(
        r=8,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    def __init__(
        self, 
        model_checkpoint: str, 
        quantization_config: BitsAndBytesConfig = None, 
        use_default_quantization_config: bool = True,
        lora_config: LoraConfig = None,
        use_lora: bool = True,
        local_files_only: bool = True,
    ):
        """
        Args
            - model_checkpoint                  : Huggingface model checkpoint
            - quantization_config               : Quantization config to use
            - use_default_quantization_config   : Whether to use the default quantization config or not
            - lora_config                       : Lora config to use
            - use_lora                          : Whether to load the model with lora or not
            - local_files_only                  : Whether the model is local or not
        """
        self.model_checkpoint = model_checkpoint
        
        self.quantization_config = None
        if quantization_config is None:
            if use_default_quantization_config:
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
        else:
            self.quantization_config = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            local_files_only=local_files_only,
            quantization_config=self.quantization_config,
            trust_remote_code=True
        )
        
        print("\n====================================================================\n")
        print("\t\t\tFETCHED MODEL")
        print("\n====================================================================\n")
        
        if use_lora and lora_config is None:
            self.lora_config = TrainingModelLoader.DEFAULT_LORA_CONFIG
        elif use_lora:
            self.lora_config = lora_config

        # self.model.config.use_cache=False
        self.model.config.pretraining_tp=1
        self.model.gradient_checkpointing_enable()
        
        if use_default_quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)

            print("\n====================================================================\n")
            print("\t\t\tAPPLIED QUANTIZATION CONFIG")       
            print("\n====================================================================\n")

        if use_lora:
            self.model = get_peft_model(self.model, self.lora_config)
        
            print("\n====================================================================\n")
            print("\t\t\tPREPARED MODEL FOR FINETUNING")       
            print("\n====================================================================\n")

            
# DEFAULT_TRAINING_ARGUMENTS = TrainingArguments(
#     output_dir='tmp',
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     optim='adamw_torch',
#     learning_rate=2e-4,
#     lr_scheduler_type='cosine',
#     save_strategy='epoch',
#     logging_steps=50,
#     # save_steps=500
#     num_train_epochs=3.0,
#     # max_steps=300,
#     fp16=True,
#     report_to="none"
# )

class ModelTrainer:
    """
    Trains a model on a dataset using Supervised Fine Tuning
    """

    def __init__(
        self, 
        dataset, 
        tokenizer_loader: TokenizerLoader, 
        model_loader: TrainingModelLoader, 
        out_dir: str, 
        training_arguments = None,
        batch_size = 2,
        **kwargs
    ):
        self.dataset = dataset
        self.tokenizer_loader = tokenizer_loader
        self.model_loader = model_loader
        self.out_dir = out_dir
        self.batch_size = batch_size
        
        self.training_arguments = training_arguments

        self.trainer = SFTTrainer(
            model=model_loader.model,
            train_dataset=dataset,
            peft_config=model_loader.lora_config,
            args=self.training_arguments,
            tokenizer=tokenizer_loader.tokenizer,
            packing=False,
            max_seq_length=1024,
            **kwargs
        )
        
        print("\n====================================================================\n")
        print("\t\t\tPREPARED FOR FINETUNING")
        print("\n====================================================================\n")

        
    def train(self):
        self.trainer.train()
    
if __name__ == "__main__":
    
    pass
