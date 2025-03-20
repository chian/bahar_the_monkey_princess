from typing import List, Dict, Any, Set, Optional, Tuple
from custom_types import Fact
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

class FactExplorer:
    """Explores and filters facts based on relevance to queries."""
    
    def __init__(self, fact_manager):
        """Initialize with a reference to the fact manager."""
        self.fact_manager = fact_manager
        self.session_manager = fact_manager.session_manager
        
        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Cache for processed terms
        self._term_cache: Dict[str, Set[str]] = {}
    
    def _process_text(self, text: str) -> Set[str]:
        """Process text into a set of lemmatized terms, excluding stopwords."""
        if text in self._term_cache:
            return self._term_cache[text]
        
        # Tokenize and clean
        tokens = word_tokenize(text.lower())
        # Remove punctuation and stopwords, then lemmatize
        terms = {
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in string.punctuation and token not in self.stop_words
        }
        
        # Cache the result
        self._term_cache[text] = terms
        return terms
    
    def _extract_terms_from_fact(self, fact: Fact) -> Set[str]:
        """Extract all relevant terms from a fact."""
        terms = set()
        
        # Process subject, relation, and object
        terms.update(self._process_text(fact['subject']))
        terms.update(self._process_text(fact['relation']))
        terms.update(self._process_text(fact['object']))
        
        # Add terms from evidence if available
        if 'evidence' in fact and fact['evidence']:
            for evidence in fact['evidence']:
                if evidence.get('text'):
                    terms.update(self._process_text(evidence['text']))
        
        return terms
    
    def _calculate_relevance_score(self, query_terms: Set[str], fact_terms: Set[str]) -> float:
        """Calculate relevance score between query terms and fact terms."""
        if not query_terms or not fact_terms:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_terms & fact_terms)
        union = len(query_terms | fact_terms)
        
        return intersection / union if union > 0 else 0.0
    
    def filter_facts(self, query: str, min_score: float = 0.1) -> List[Tuple[Fact, float]]:
        """
        Filter facts based on relevance to query.
        Returns list of (fact, score) tuples sorted by relevance score.
        """
        if not self.session_manager.facts:
            return []
        
        # Process query terms
        query_terms = self._process_text(query)
        
        # Calculate scores for all facts
        scored_facts = []
        for fact in self.session_manager.facts:
            fact_terms = self._extract_terms_from_fact(fact)
            score = self._calculate_relevance_score(query_terms, fact_terms)
            if score >= min_score:
                scored_facts.append((fact, score))
        
        # Sort by score descending
        return sorted(scored_facts, key=lambda x: x[1], reverse=True)
    
    def explore_related(self, fact: Fact, max_distance: int = 2) -> List[Tuple[Fact, int, float]]:
        """
        Find facts related to the given fact up to max_distance steps away.
        Returns list of (fact, distance, relevance_score) tuples.
        """
        if not self.session_manager.facts:
            return []
        
        # Get terms from the source fact
        source_terms = self._extract_terms_from_fact(fact)
        
        # Track visited facts to avoid cycles
        visited = {self.fact_manager._fact_to_key(fact)}
        related_facts = []
        
        def explore_level(current_fact: Fact, current_distance: int):
            if current_distance > max_distance:
                return
            
            # Look for facts sharing terms with current fact
            current_terms = self._extract_terms_from_fact(current_fact)
            
            for candidate in self.session_manager.facts:
                candidate_key = self.fact_manager._fact_to_key(candidate)
                if candidate_key in visited:
                    continue
                
                # Calculate relevance to both source and current fact
                candidate_terms = self._extract_terms_from_fact(candidate)
                source_score = self._calculate_relevance_score(source_terms, candidate_terms)
                current_score = self._calculate_relevance_score(current_terms, candidate_terms)
                
                # Use combined score weighted by distance
                combined_score = (source_score + current_score) / (2 * current_distance)
                
                if combined_score > 0:
                    visited.add(candidate_key)
                    related_facts.append((candidate, current_distance, combined_score))
                    
                    # Explore from this fact if score is good enough
                    if combined_score >= 0.2:  # Threshold for further exploration
                        explore_level(candidate, current_distance + 1)
        
        # Start exploration from the source fact
        explore_level(fact, 1)
        
        # Sort by score descending
        return sorted(related_facts, key=lambda x: x[2], reverse=True)
    
    def print_filtered_facts(self, facts_with_scores: List[Tuple[Fact, float]]) -> None:
        """Print facts with their relevance scores in a readable format."""
        if not facts_with_scores:
            print("No relevant facts found.")
            return
        
        print(f"\nFound {len(facts_with_scores)} relevant facts:")
        for i, (fact, score) in enumerate(facts_with_scores, 1):
            # Format the basic fact information
            fact_str = f"{fact['subject']} {fact['relation']} {fact['object']}"
            
            # Add verification status if available
            status = ""
            if 'verification' in fact:
                status = f"[{fact['verification'].get('status', 'unknown').upper()}]"
            
            # Add source if available
            source = ""
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
                source = f"(Source: {source_id})"
            
            print(f"{i}. {status} {fact_str} {source}")
            print(f"   Relevance: {score:.2f}")
            
            # Add evidence snippet if available
            if 'evidence' in fact and fact['evidence'] and fact['evidence'][0].get('text'):
                evidence_text = fact['evidence'][0]['text']
                if len(evidence_text) > 100:
                    evidence_text = evidence_text[:97] + "..."
                print(f"   Evidence: {evidence_text}")
            
            print()  # Empty line between facts
    
    def print_related_facts(self, related_facts: List[Tuple[Fact, int, float]]) -> None:
        """Print related facts with their distances and scores in a readable format."""
        if not related_facts:
            print("No related facts found.")
            return
        
        print(f"\nFound {len(related_facts)} related facts:")
        for i, (fact, distance, score) in enumerate(related_facts, 1):
            # Format the basic fact information
            fact_str = f"{fact['subject']} {fact['relation']} {fact['object']}"
            
            # Add verification status if available
            status = ""
            if 'verification' in fact:
                status = f"[{fact['verification'].get('status', 'unknown').upper()}]"
            
            # Add source if available
            source = ""
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
                source = f"(Source: {source_id})"
            
            print(f"{i}. {status} {fact_str} {source}")
            print(f"   Distance: {distance}, Relevance: {score:.2f}")
            
            # Add evidence snippet if available
            if 'evidence' in fact and fact['evidence'] and fact['evidence'][0].get('text'):
                evidence_text = fact['evidence'][0]['text']
                if len(evidence_text) > 100:
                    evidence_text = evidence_text[:97] + "..."
                print(f"   Evidence: {evidence_text}")
            
            print()  # Empty line between facts 