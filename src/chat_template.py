class ChatTemplate:
    """
    Applies a chat template to a set of input
    """
    def __init__(self, tokenizer, system_entry: str = None, starting_user_entry: str = None):
        """
        Args
            - tokenizer             : Tokenizer used in the model
            - system_entry          : Base system entry
            - starting_user_entry   : Starting user entry
        """
        self.tokenizer = tokenizer
        self.messages = []
        
        if system_entry:
            self.add_system_entry(system_entry)
            
        if starting_user_entry:
            self.add_user_entry(starting_user_entry)
    
    def reset(self):
        """
        Removes all messages in the template
        """
        self.messages = []
    
    def add_entry(self, role: str, entry: str, add_generation_prompt=True):
        """
        Add an entry based on a role

        Args
            - role                      : Role associated to the entry
            - entry                     : Entry to add
            - add_generation_prompt     : Whether to add the generation prompt after the entry
        """
        self.messages.append({'role': role, 'content': entry})
        return self.apply(self.messages, add_generation_prompt=add_generation_prompt)
    
    def single_user_entry(self, entry: str):
        return self.apply([{'role': 'user', 'content': entry}])
    
    def add_user_entry(self, entry: str, add_generation_prompt=True):
        return self.add_entry('user', entry, add_generation_prompt)
        
    def add_assistant_entry(self, entry: str, add_generation_prompt=False):
        return self.add_entry('assistant', entry, add_generation_prompt)
        
    def add_system_entry(self, entry: str, add_generation_prompt=True):
        return self.add_entry('system', entry, add_generation_prompt)
    
    def apply(self, messages: list = None, add_generation_prompt=True):
        """
        Applies a chat template to a list of messages

        Args
            - messages                  : List of messages
            - add_generation_prompt     : Whether to add the generation prompt after the entry
        """
        if messages is None:
            return self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=add_generation_prompt, return_tensors="pt")
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, return_tensors="pt")
