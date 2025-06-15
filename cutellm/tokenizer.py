"""
Simple tokenizer implementation for CuteLLM.
"""
import torch
import logging

class SimpleWordTokenizer:
    """A basic word-level tokenizer for CuteLLM
    
    This tokenizer provides simple word-level tokenization for educational purposes.
    It includes a fixed vocabulary of common English words and special tokens.
    """
    def __init__(self, vocab_size=1000):
        logging.info(f"Initializing SimpleWordTokenizer with vocab_size={vocab_size}")
        self.vocab_size = vocab_size
        
        # Create a vocabulary with common words
        logging.info("Creating vocabulary with common words")
        self.common_words = [
            # Common nouns
            "cat", "dog", "house", "car", "book", "tree", "table", "chair", "floor", "bed",
            "computer", "phone", "food", "water", "door", "window", "room", "kitchen", "garden",
            "school", "store", "park", "home", "ball", "toy", "game", "sun", "moon", "sky",
            
            # Common verbs
            "is", "are", "was", "were", "be", "have", "has", "had", "do", "does", "did",
            "go", "goes", "went", "see", "saw", "eat", "ate", "run", "ran", "walk", "walked",
            "sit", "sat", "stand", "stood", "sleep", "slept", "talk", "talked", "say", "said",
            
            # Common adjectives
            "big", "small", "good", "bad", "hot", "cold", "new", "old", "happy", "sad",
            "fast", "slow", "hard", "soft", "loud", "quiet", "clean", "dirty", "light", "dark",
            
            # Common prepositions
            "in", "on", "at", "by", "with", "from", "to", "for", "of", "about",
            
            # Common determiners and pronouns
            "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her",
            "our", "their", "it", "they", "we", "i", "you", "he", "she", "them", "us",
            
            # Common adverbs
            "very", "really", "quite", "too", "also", "only", "just", "now", "then", "here",
            "there", "always", "never", "sometimes", "often", "again"
        ]
        
        # Create word-to-id and id-to-word mappings
        logging.info("Creating word-to-id mapping")
        self.word_to_id = {word: i for i, word in enumerate(self.common_words)}
        
        # Add special tokens
        logging.info("Adding special tokens: <unk>, <mask>")
        self.word_to_id["<unk>"] = len(self.common_words)  # Unknown token
        self.word_to_id["<mask>"] = len(self.common_words) + 1  # Mask token
        
        logging.info("Creating id-to-word mapping")
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        
        # Store mask token ID for easy access
        self.mask_token_id = self.word_to_id["<mask>"]
        
        actual_vocab_size = len(self.word_to_id)
        logging.info(f"Tokenizer initialized with {actual_vocab_size} tokens ({len(self.common_words)} common words + 2 special tokens)")
        
    def encode(self, text):
        """Convert text to token IDs"""
        logging.debug(f"Encoding text: '{text[:50]}{'...' if len(text) > 50 else ''}' ({len(text)} chars)")
        
        # Replace ___ with <mask> token for phrase completion tasks
        text = text.replace("___", "<mask>")
        logging.debug("Replaced ___ placeholders with <mask> token")
        
        # Split text into words and convert to lowercase
        words = text.lower().split()
        logging.debug(f"Split into {len(words)} words")
        
        # Convert words to token IDs
        token_ids = [self.word_to_id.get(word, self.word_to_id["<unk>"]) for word in words]
        
        # Count unknown tokens
        unknown_count = sum(1 for id in token_ids if id == self.word_to_id["<unk>"])
        if unknown_count > 0:
            logging.debug(f"Found {unknown_count} unknown words ({unknown_count/len(words):.1%} of total)")
            
        logging.debug(f"Encoded to {len(token_ids)} token IDs")
        return token_ids
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        logging.debug(f"Decoding {len(ids)} token IDs")
        text = " ".join([self.id_to_word.get(id, "<unk>") for id in ids])
        logging.debug(f"Decoded to text: '{text[:50]}{'...' if len(text) > 50 else ''}' ({len(text)} chars)")
        return text
    
    def get_mask_position(self, ids):
        """Find the position of the mask token in the input"""
        logging.debug(f"Searching for mask token (ID: {self.mask_token_id}) in sequence of length {len(ids)}")
        try:
            pos = ids.index(self.mask_token_id)
            logging.debug(f"Found mask token at position {pos}")
            return pos
        except ValueError:
            logging.debug("No mask token found in sequence")
            return -1
            
    def get_word_from_id(self, id):
        """Get the word corresponding to an ID"""
        word = self.id_to_word.get(id, "<unk>")
        logging.debug(f"Token ID {id} corresponds to word '{word}'")
        return word
    
    def get_vocabulary_size(self):
        """Get the actual size of the vocabulary"""
        vocab_size = len(self.word_to_id)
        logging.debug(f"Vocabulary size: {vocab_size}")
        return vocab_size
