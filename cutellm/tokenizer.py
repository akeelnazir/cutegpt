"""
Simple tokenizer implementation for CuteLLM.
"""
import torch

class SimpleWordTokenizer:
    """A basic word-level tokenizer for CuteLLM
    
    This tokenizer provides simple word-level tokenization for educational purposes.
    It includes a fixed vocabulary of common English words and special tokens.
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        
        # Create a vocabulary with common words
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
        self.word_to_id = {word: i for i, word in enumerate(self.common_words)}
        
        # Add special tokens
        self.word_to_id["<unk>"] = len(self.common_words)  # Unknown token
        self.word_to_id["<mask>"] = len(self.common_words) + 1  # Mask token
        
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        
        # Store mask token ID for easy access
        self.mask_token_id = self.word_to_id["<mask>"]
        
    def encode(self, text):
        """Convert text to token IDs"""
        # Replace ___ with <mask> token for phrase completion tasks
        text = text.replace("___", "<mask>")
        # Split text into words and convert to lowercase
        words = text.lower().split()
        # Convert words to token IDs
        return [self.word_to_id.get(word, self.word_to_id["<unk>"]) for word in words]
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        return " ".join([self.id_to_word.get(id, "<unk>") for id in ids])
    
    def get_mask_position(self, ids):
        """Find the position of the mask token in the input"""
        try:
            return ids.index(self.mask_token_id)
        except ValueError:
            return -1
            
    def get_word_from_id(self, id):
        """Get the word corresponding to an ID"""
        return self.id_to_word.get(id, "<unk>")
    
    def get_vocabulary_size(self):
        """Get the actual size of the vocabulary"""
        return len(self.word_to_id)
