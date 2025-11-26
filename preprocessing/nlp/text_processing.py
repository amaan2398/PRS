# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS
import re
import unicodedata
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import spacy
from bs4 import BeautifulSoup
import contractions
from spellchecker import SpellChecker

from preprocessing.nlp import NlpEngine
from config.manager import ConfigManager

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

class TextProcessor:
    """Handles text normalization using SpaCy's efficient pipeline."""

    def __init__(self):
        self.nlp, self.spell = NlpEngine.get_spacy()
    
    def _nlp_clean_text(self, text: str, basic_cleaning: bool = True) -> str:
        """
        Optimized text cleaning for sentiment models or BoW/TF-IDF.
        Includes: HTML removal, URLs, contractions, stopwords, lemmatization,
        and optimized spell-correction using pyspellchecker.
        """
        if not isinstance(text, str):
            return ""
        
        if basic_cleaning:
            # 1. Lowercase
            text = text.lower()

        # 2. Remove HTML
        if bool(re.search(r'<[^>]+>', text)):
            text = BeautifulSoup(text, "lxml").get_text()

        # 3. Expand contractions
        text = contractions.fix(text)
        
        # 4. Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # 5. Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # 6. Remove @mentions and hashtags
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)

        # 7. Remove standalone numbers
        text = re.sub(r"\b\d+\b", "", text)

        # 8. Remove special chars (keep alphabets)
        text = re.sub(r"[^a-z\s]", "", text)

        # ---- Tokenization ----
        doc = self.nlp(text)

        # Collect tokens after cleaning (pre-spellcheck)
        tokens = []

        for token in doc:
            if token.is_stop:
                continue
            if token.like_num:
                continue
            if len(token.text) < 3:  # skip tiny words
                continue

            lemma = token.lemma_.strip()
            if lemma:
                tokens.append(lemma)

        # ---- Optimized Spell Correction ----
        # 1. Get unknown words only
        unknown_words = self.spell.unknown(tokens)

        corrected_tokens = []
        for token in tokens:
            if token not in unknown_words:
                corrected_tokens.append(token)
            else:
                if len(token) >= 4:
                    corrected = self.spell.correction(token)
                    # FIX: pyspellchecker may return None
                    if corrected is None:
                        corrected = token
                    corrected_tokens.append(corrected)
                else:
                    corrected_tokens.append(token)

        # Return final text
        return " ".join(corrected_tokens)

    def process_batch(self, texts: List[Optional[str]], batch_size: int = 2000, nlp_text_cleaner: bool = False) -> List[Optional[str]]:
        """
        Processes a list of texts using SpaCy's pipe method.
        
        Args:
            texts: List of raw strings.
            batch_size: Number of docs to process at once.
            
        Returns:
            List of cleaned strings.
        """
        # 1. Pre-filter inputs to avoid sending None/Empty strings to SpaCy
        valid_indices = []
        valid_inputs = []

        # Initialize result list with None
        cleaned_results = [None] * len(texts)
        
        for idx, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                # Basic unicode normalization (fast)
                norm_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
                valid_inputs.append(norm_text.lower())
                valid_indices.append(idx)

        if not valid_inputs:
            return cleaned_results

        print(f"Processing {len(valid_inputs)} valid text entries via SpaCy pipe...")

        if nlp_text_cleaner:
            if HAS_TQDM:
                valid_inputs = tqdm(valid_inputs, total=len(valid_inputs), desc="NLP Text Normalization")

            for idx, text in zip(valid_indices, valid_inputs):
                cleaned_results[idx] = self._nlp_clean_text(text)
        else:
            # 2. Process in batches
            doc_stream = self.nlp.pipe(valid_inputs, batch_size=batch_size)
            
            if HAS_TQDM:
                doc_stream = tqdm(doc_stream, total=len(valid_inputs), desc="SpaCy Text Normalization")

            # 3. Reconstruct results
            for doc, original_idx in zip(doc_stream, valid_indices):
                processed_tokens = []
                
                for token in doc:
                    # Filter out punctuation, symbols, and whitespace
                    if not (token.is_punct or token.is_space or token.is_stop):
                        processed_tokens.append(token.text)
                
                # Join back into a string
                if processed_tokens:
                    cleaned_results[original_idx] = " ".join(processed_tokens)
                else:
                    cleaned_results[original_idx] = ""

        return cleaned_results
    
    @staticmethod
    def clean_and_split_str(text: Optional[str], delimiter: str = ',', replacements: List[Tuple[str, str]] = []) -> List[str]:
        """
        This function takes a string and splits it into a list of strings based on the delimiter.
        It also applies a list of replacements to each string in the list.

        Args:
            text (Optional[str]): The string to be cleaned and split.
            delimiter (str, optional): The delimiter to split the string on. Defaults to ','.
            replacements (List[Tuple[str, str]], optional): A list of tuples containing the pattern and replacement to apply to each string. Defaults to [].

        Returns:
            List[str]: A list of cleaned, standardized category names.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        # Optimization: Use list comprehension for efficient string manipulation
        replacements = [(rep['pattern'], rep['replacement'])for rep in replacements]
        cleaned_list = [re.sub(pattern, replacement, c).strip() for pattern, replacement in replacements for c in text.lower().split(delimiter)]

        
        # Filter out any resulting empty strings
        return [c for c in cleaned_list if c]


class StandardizeNameData:
    """Standardizes brand/manufacturer names using Regex rules from ConfigManager."""

    def __init__(self, target_column: str = "brand") -> None:
        self.config_manager = ConfigManager()
        self.all_text_config = self.config_manager.get_config("text_processing")
        
        if not self.all_text_config:
            print("Warning: 'text_processing' configuration not found.")
            self.direct_map = {}
            self.heuristic_rules = []
        else:
            # Fetch config for the specific column, defaulting to empty if not found
            column_config = self.all_text_config.get(target_column, {})
            self.direct_map = column_config.get("direct_map", {})
            self.heuristic_rules = column_config.get("heuristic_rules", [])
            
            if not column_config:
                print(f"Warning: No configuration found for column '{target_column}'. Using empty rules.")

        # Compile regex once upon instantiation
        self._compiled_heuristics = [
            (re.compile(pattern), replacement) 
            for pattern, replacement in self.heuristic_rules
        ]

    def standardize(self, text: Optional[str]) -> Optional[str]:
        """Applies direct mapping and regex rules to a single string."""
        if not isinstance(text, str) or not text:
            return None
        
        text_clean = text.strip().lower()

        # 1. Exact Match
        if text_clean in self.direct_map:
            return self.direct_map[text_clean]

        # 2. Regex Patterns
        for pattern, replacement in self._compiled_heuristics:
            if pattern.search(text_clean):
                return replacement

        return text_clean

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Applies standardization to a Pandas Series."""
        if HAS_TQDM:
            tqdm.pandas(desc="Standardizing Brands")
            return series.progress_apply(self.standardize)
        return series.apply(self.standardize)
