from spellchecker import SpellChecker
import spacy

class NlpEngine:
    """Singleton wrapper for SpaCy to ensure the model is loaded only once."""
    
    _spacy_model = None
    _spell = None

    @classmethod
    def get_spacy(cls):
        """Loads SpaCy model. Disables heavy components (parser, NER) for speed."""
        if cls._spacy_model is None:
            print("Loading SpaCy model (en_core_web_sm)...")
            try:
                # We only need the tokenizer, disable the rest for max speed
                cls._spacy_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                raise OSError(
                    "SpaCy model 'en_core_web_sm' not found. "
                    "Please run: python -m spacy download en_core_web_sm"
                )
        if cls._spell is None:
            cls._spell = SpellChecker()
        return cls._spacy_model, cls._spell