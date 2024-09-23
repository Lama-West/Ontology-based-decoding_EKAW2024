from src.generation import DomainOntologyBasedVerbalizer

# In general, the prompts are divided into two sections : `pre` and `post`. `Pre` corresponds to the prompt before the clinical
# note and `post` corresponds to the content after the clinical note


# ================================= Domain Adaptation =================================

# Prompts used for greedy search and diverse beam search
DOMAIN_NORMAL_PRE_PROMPT = f"""Here are some clinical notes. Every sequence of '=' indicates a different note about the same patient made by a different clinician : \n"""
DOMAIN_NORMAL_POST_PROMPT = '\n===========\n\n' + "Summarize these clinical notes in a short text by focusing on concepts related to the " + DomainOntologyBasedVerbalizer.DOMAIN_HEADER + " domain."

# Prompts used for our method (without specifying the domain)
DOMAIN_OURS_PRE_PROMPT = "Here are some clinical notes that were structured by concepts. Every sequence of '=' indicates a different note about the same patient made by a different clinician : \n"
DOMAIN_OURS_POST_PROMPT = "\n===========\n\n" + "Summarize these clinical notes in a short text."

# ====================================== BHC Task ======================================

BHC_FINAL_SENTENCE = "In a short text, summarize the events occurring to the patient during the hospital stay, the surgical, medical and other consults the patient experienced and the hospital procedures the patient experienced."
BHC_STRUCTURED_INPUT = "Here are a patient's clinical notes organized as a series of key-value pairs. Keys represent medical concepts and values provide specific details, observations, or interpretations about the patient related to the key."

BHC_NORMAL_PRE_PROMPT = "Here are a patient's clinical notes separated by sequences of '='. \n===========\\n"
BHC_NORMAL_POST_PROMPT = "\n===========\n\n" + BHC_FINAL_SENTENCE

BHC_EXT_PRE_PROMPT = BHC_STRUCTURED_INPUT + "\n\n"
BHC_EXT_POST_PROMPT = "\n\n" + BHC_FINAL_SENTENCE

BHC_EXT_PRU_PRE_PROMPT = BHC_STRUCTURED_INPUT + "\n\n"
BHC_EXT_PRU_POST_PROMPT = "\n\n" + BHC_FINAL_SENTENCE

# =================================== Extraction step ===================================

BASE_PROMPT_TEMPLATE="""Here is a clinical note about a patient : 
-------------------
{clinical_note}
-------------------
In a short sentence, summarize everything related to the "{label}" concept mentioned the clinical note. {properties}. \n\n If the concept is not mentioned in the note, respond with 'N/A'.
"""
