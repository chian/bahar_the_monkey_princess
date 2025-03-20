# Import the necessary dependencies
import hashlib
import json
import time
from query_atom_smallworld import DEFAULT_CONFIG, FIRECRAWL_CONFIG, SCIENTIFIC_RELATIONS
from firecrawl import FirecrawlApp
import openai
from typing import Dict, List, Any, Optional
from custom_types import Fact, Verification, Evidence  # Updated import

class FactVerifier:
    """Handles verification of facts using external sources."""
    
    def __init__(self, fact_manager: 'FactManager'):
        """Initialize with a reference to the fact manager."""
        self.fact_manager = fact_manager
        self.session_manager = fact_manager.session_manager
        self.openai_client = None
        self.firecrawl_app = None
        self.scientific_relations = SCIENTIFIC_RELATIONS

    def verify_facts(self, cli_instance, source_id: str) -> None:
        """
        Verify facts from a specific source using Firecrawl search.
        Facts will be classified as 'confirmed', 'unverified', 'false', or 'controversial'.
        
        Usage: verify_facts <source_id>
        """
        if not cli_instance.facts:
            print("No facts available. Start a research session first.")
            return
        
        if not source_id:
            print("Please specify a source ID to verify facts from.")
            print("Use 'list_sources' to see available sources.")
            return
        
        # Find facts with the specified source and track which ones need verification
        unverified_facts = []
        already_verified = 0
        
        for fact in cli_instance.facts:
            source_id_from_fact = "unknown"
            if 'evidence' in fact and fact['evidence']:
                source_id_from_fact = fact['evidence'][0].get('source_id', "unknown")
                
            if source_id_from_fact == source_id:
                if 'verification' in fact and fact['verification'].get('status') in ['confirmed', 'unverified', 'false', 'controversial']:
                    already_verified += 1
                else:
                    unverified_facts.append(fact)
        
        total_facts = already_verified + len(unverified_facts)
        
        if not unverified_facts:
            if already_verified > 0:
                print(f"All facts from source '{source_id}' have already been verified ({already_verified} facts).")
                print("Use 'facts_by_source' to see the verified facts.")
            else:
                print(f"No facts found from source '{source_id}'.")
            return
        
        print(f"Found {len(unverified_facts)} facts from source '{source_id}' to verify.")
        print("This process may take some time as it involves web searches and analysis.")
        
        confirmation = input("Do you want to proceed with verification? (y/n): ")
        if confirmation.lower() != 'y':
            print("Verification cancelled.")
            return
        
        # Import necessary functions from query_atom_smallworld
        try:
            if not hasattr(cli_instance, '_openai_client'):
                cli_instance._openai_client = openai.OpenAI(api_key=DEFAULT_CONFIG["api_key"])
                
            firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_CONFIG["api_key"])
            
            # Set up the verifier properties
            self.openai_client = cli_instance._openai_client
            self.firecrawl_app = firecrawl_app
            self.scientific_relations = SCIENTIFIC_RELATIONS
            
            # Set up for tracking verification results
            verification_results = {
                'confirmed': 0,   # Counter for confirmed facts
                'unverified': 0,  # Counter for unverified facts
                'false': 0,       # Counter for false facts
                'controversial': 0  # Counter for controversial facts
            }
            
            # Process each fact
            for i, fact in enumerate(unverified_facts, 1):
                print(f"\nVerifying fact {i}/{len(unverified_facts)} (total progress: {already_verified + i}/{total_facts}):")
                subject = fact.get('subject', '')
                relation = fact.get('relation', '')
                obj = fact.get('object', '')
                print(f"  {subject} {relation} {obj}")
                
                # Generate search queries for this fact
                search_queries = self._generate_search_queries(fact)
                
                # Initialize verification data
                verification_data: Dict[str, Any] = {
                    'fact': fact,
                    'searches': [],
                    'confirming_texts': [],
                    'contradicting_texts': [],
                    'status': 'unverified'
                }
                
                # Use our patched function that uses the correct parameters
                print(f"  Searching with Firecrawl using {len(search_queries)} queries...")
                search_results = self.verify_search(search_queries[0])
                
                if not search_results:
                    print("  No search results found.")
                    verification_data['status'] = 'unverified'
                    verification_results['unverified'] += 1
                    continue
                
                print(f"  Found {len(search_results)} total results")
                
                # Track search information
                for query in search_queries:
                    verification_data['searches'].append({
                        'query': query,
                        'results_count': len([r for r in search_results if r.get('query') == query])
                    })
                
                # Analyze each search result
                for result in search_results:
                    try:
                        content = result.get('content', '')
                        if not content:
                            continue
                        
                        # Initialize analysis result
                        analysis_result = {
                            'confirms': False,
                            'contradicts': False,
                            'confidence': 0.0,
                            'relevant_text': None,
                            'new_facts': []
                        }
                        
                        # Update result if we found better evidence
                        if content and (not analysis_result['relevant_text'] or len(content) > len(analysis_result['relevant_text'])):
                            analysis_result['relevant_text'] = content
                            analysis_result['confirms'] = True
                        
                        if analysis_result['confirms']:
                            verification_data['confirming_texts'].append({
                                'url': result.get('url', ''),
                                'title': result.get('title', ''),
                                'text': analysis_result['relevant_text']
                            })
                        else:
                            verification_data['contradicting_texts'].append({
                                'url': result.get('url', ''),
                                'title': result.get('title', ''),
                                'text': analysis_result['relevant_text']
                            })
                        
                        # Also extract new facts if possible
                        if analysis_result['new_facts']:
                            for new_fact in analysis_result['new_facts']:
                                print(f"  [NEW FACT] Extracted new fact: {new_fact['subject']} {new_fact['relation']} {new_fact['object']}")
                                # Add the new fact with source
                                new_fact['evidence'] = [{
                                    'source_id': f"web_{hashlib.md5(result.get('url', '').encode()).hexdigest()[:8]}",
                                    'url': result.get('url', ''),
                                    'title': result.get('title', ''),
                                    'search_query': result.get('query', '')
                                }]
                                cli_instance.facts.append(new_fact)
                                
                                # Also add to research data
                                if cli_instance.research_data and 'facts' in cli_instance.research_data:
                                    cli_instance.research_data['facts'].append(new_fact)
                
                    except Exception as e:
                        print(f"  Error during verification: {e}")
                        # Save the fact as unverified
                        fact['verification'] = {
                            'status': 'unverified',
                            'confirming': 0,
                            'contradicting': 0,
                            'error': str(e)
                        }
                        verification_results['unverified'] += 1
                
                # Determine final verification status
                if verification_data['confirming_texts'] and verification_data['contradicting_texts']:
                    verification_data['status'] = 'controversial'
                    print("  [WARNING] Fact is CONTROVERSIAL - found both confirming and contradicting evidence")
                    verification_results['controversial'] += 1
                    
                    # Store the original source ID before updating anything
                    original_source_id = fact['evidence'][0].get('source_id', 'unknown') if 'evidence' in fact and fact['evidence'] else 'unknown'
                    
                    # Update the fact with verification status
                    verification: Verification = {
                        'status': 'controversial',
                        'confirming': len(verification_data['confirming_texts']),
                        'contradicting': len(verification_data['contradicting_texts']),
                        'evidence': verification_data['confirming_texts'] + verification_data['contradicting_texts'],
                        'original_source_id': original_source_id,
                        'error': None
                    }
                    fact['verification'] = verification
                    
                    # Update the fact's primary source to the confirming source
                    if 'evidence' in fact and fact['evidence']:
                        # Store the URL from the first confirming evidence as the new source_id
                        new_source_id = verification_data['confirming_texts'][0].get('url', '')
                        
                        # Update the evidence with the new source info
                        fact['evidence'][0] = Evidence(
                            source_id=new_source_id,
                            url=new_source_id,
                            title=verification_data['confirming_texts'][0].get('title'),
                            text=verification_data['confirming_texts'][0].get('text'),
                            chunk_id=None,
                            search_query=None,
                            content=None,
                            original_source_id=original_source_id
                        )
                        
                        print(f"  Updated fact source from '{original_source_id}' to '{new_source_id}'")
                        
                        # Also update the fact in research_data if it exists there
                        if cli_instance.research_data and 'facts' in cli_instance.research_data:
                            # Find the same fact in research_data and update it
                            for research_fact in cli_instance.research_data['facts']:
                                if (research_fact.get('subject') == fact.get('subject') and
                                    research_fact.get('relation') == fact.get('relation') and
                                    research_fact.get('object') == fact.get('object')):
                                    # Update the verification and evidence in the research_data copy
                                    research_fact['verification'] = verification
                                    research_fact['evidence'] = fact['evidence']
                                    break
                
                elif verification_data['contradicting_texts'] and not verification_data['confirming_texts']:
                    verification_data['status'] = 'false'
                    verification_results['false'] += 1
                    
                    # Display the actual contradicting evidence
                    print(f"  [FALSE] Fact is FALSE - found contradicting evidence:")
                    for i, text in enumerate(verification_data['contradicting_texts'][:3], 1):  # Limit to first 3 for brevity
                        # If 'content' or 'text' is in the dictionary, use that, otherwise use the string itself
                        if isinstance(text, dict):
                            content = text.get('content', text.get('text', str(text)))
                            source = text.get('url', text.get('source', 'unknown source'))
                        else:
                            content = str(text)
                            source = 'unknown source'
                        
                        # Truncate content if it's too long
                        if len(content) > 200:
                            content = content[:197] + "..."
                        
                        print(f"    {i}. {content}")
                        print(f"       Source: {source}")
                    
                    # Use fact_manager to delete the fact instead of direct removal
                    fact_index = self.session_manager.facts.index(fact)
                    self.fact_manager.delete_fact(fact_index)
                    print("  Automatically deleted false fact.")
                
                elif verification_data['confirming_texts']:
                    verification_data['status'] = 'confirmed'
                    print("  [CONFIRMED] Fact is CONFIRMED")
                    verification_results['confirmed'] += 1
                    
                    # Sort by text length for relevance
                    sorted_evidence = sorted(
                        verification_data['confirming_texts'],
                        key=lambda x: len(x.get('text', '')) if x.get('text') else 0,
                        reverse=True
                    )
                    
                    if sorted_evidence:
                        best_evidence = sorted_evidence[0]
                    else:
                        best_evidence = {'url': '', 'title': '', 'text': ''}
                    
                    # Update the fact with verification status
                    verification: Verification = {
                        'status': 'confirmed',
                        'confirming': 1,
                        'contradicting': 0,
                        'evidence': [{
                            'url': best_evidence.get('url', ''),
                            'title': best_evidence.get('title', ''),
                            'text': best_evidence.get('text', '')
                        }],
                        'original_source_id': None,
                        'error': None
                    }
                    fact['verification'] = verification
                    
                    # Update the fact's primary source to the confirming source
                    if 'evidence' in fact and fact['evidence']:
                        # Store the URL from the best evidence as the new source_id
                        new_source_id = best_evidence.get('url', '')
                        
                        # Update the evidence with the new source info
                        fact['evidence'][0] = Evidence(
                            source_id=new_source_id,
                            url=new_source_id,
                            title=best_evidence.get('title'),
                            text=best_evidence.get('text'),
                            chunk_id=None,
                            search_query=None,
                            content=None,
                            original_source_id=None
                        )
                        
                        print(f"  Updated fact source from 'None' to '{new_source_id}'")
                        
                        # Also update the fact in research_data if it exists there
                        if cli_instance.research_data and 'facts' in cli_instance.research_data:
                            # Find the same fact in research_data and update it
                            for research_fact in cli_instance.research_data['facts']:
                                if (research_fact.get('subject') == fact.get('subject') and
                                    research_fact.get('relation') == fact.get('relation') and
                                    research_fact.get('object') == fact.get('object')):
                                    # Update the verification and evidence in the research_data copy
                                    research_fact['verification'] = verification
                                    research_fact['evidence'] = fact['evidence']
                                    break
                
                else:
                    verification_data['status'] = 'unverified'
                    print("  [UNVERIFIED] Fact is UNVERIFIED - could not find confirming or contradicting evidence")
                    verification_results['unverified'] += 1
                
                # Save session after each fact to preserve progress
                cli_instance._save_session()
                print("  Progress saved.")
            
            # Print summary
            print("\nVerification Summary:")
            print(f"  [CONFIRMED] Confirmed: {verification_results['confirmed']}")
            print(f"  [UNVERIFIED] Unverified: {verification_results['unverified']}")
            print(f"  [FALSE] False: {verification_results['false']}")
            print(f"  [WARNING] Controversial: {verification_results['controversial']}")
            
            # Save the session with verification results
            cli_instance._save_session()

        except ImportError as e:
            print(f"Error importing required functions: {e}")
            return
    
    def verify_search(self, query):
        """Search with Firecrawl and return results."""
        print(f"Searching with Firecrawl: {query}")
        try:
            # Update to match V1 implementation
            response = self.firecrawl_app.search(
                query=query, 
                params={
                    'timeout': 30000,
                    'limit': 5,
                    'scrapeOptions': {'formats': ['markdown']}
                }
            )
            
            # Process results
            results = []
            
            # Handle different response structures as in V1
            if hasattr(response, 'data'):
                data_items = response.data
            elif isinstance(response, dict) and 'data' in response:
                data_items = response['data']
            else:
                print(f"Unexpected response format: {type(response)}")
                data_items = []
            
            for item in data_items:
                # Extract URL
                url = item.url if hasattr(item, 'url') else item.get('url', '')
                
                # Extract content
                content = item.markdown if hasattr(item, 'markdown') else item.get('markdown', '')
                if not content:
                    content = item.content if hasattr(item, 'content') else item.get('content', '')
                
                # Get title
                title = item.title if hasattr(item, 'title') else item.get('title', url)
                
                results.append({
                    "title": title,
                    "url": url,
                    "content": content
                })
            
            return results
        except Exception as e:
            print(f"Error searching with Firecrawl: {e}")
            return []
            
    def _generate_search_queries(self, fact):
        """Generate search queries for a fact, starting with specific and moving to general."""
        subject = fact.get('subject', '')
        relation = fact.get('relation', '')
        obj = fact.get('object', '')
        
        # Start with a specific query combining all elements
        queries = [
            f"{subject} {relation} {obj}",
        ]
        
        # Add more general queries
        if len(subject) > 3 and len(obj) > 3:
            queries.append(f"{subject} {obj}")
        
        # For specific types of relations, create custom queries
        if relation == "has_property":
            queries.append(f"is {subject} {obj}")
            queries.append(f"{subject} is {obj}")
        elif relation == "causes":
            queries.append(f"{subject} causes {obj}")
            queries.append(f"does {subject} cause {obj}")
        elif relation == "inhibits":
            queries.append(f"{subject} inhibits {obj}")
            queries.append(f"{subject} prevents {obj}")
        elif relation == "interacts_with":
            queries.append(f"{subject} interacts with {obj}")
            queries.append(f"{subject} {obj} interaction")
        
        # Add simple fact query for last resort
        if len(queries) < 3:
            queries.append(f"{subject}")
            queries.append(f"{obj}")
        
        return queries

    def _analyze_content_for_fact(self, fact, content):
        """
        Analyze if the content confirms or contradicts the fact.
        Also extract any new facts found in the content.
        """
        subject = fact.get('subject', '')
        relation = fact.get('relation', '')
        obj = fact.get('object', '')
        
        # Default result structure
        result = {
            'confirms': False,
            'contradicts': False,
            'confidence': 0.0,
            'relevant_text': "",
            'new_facts': []
        }
        
        try:
            # Get chunk parameters from agent config
            agent_config = AGENT_CONFIGS["information_extraction"]
            max_chunk_size = agent_config.get("max_chunk_size", 12000)
            max_chunks = agent_config.get("max_chunks", 3)
            
            # Split content into paragraphs and process in chunks
            paragraphs = content.split('\n\n')
            current_chunk = ""
            chunks = []
            
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                    current_chunk += para + '\n\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = para + '\n\n'
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Process each chunk up to max_chunks
            for i, chunk in enumerate(chunks[:max_chunks]):
                if i > 0:
                    print(f"  Analyzing content chunk {i+1}/{min(len(chunks), max_chunks)}...")
                
                # Use LLM to analyze if the content confirms, contradicts, or is neutral about the fact
                prompt = f"""Analyze if the following text confirms, contradicts, or is neutral about this scientific fact:

Fact: {subject} {relation} {obj}

Text: {chunk}

Please respond in JSON format:
{{
    "confirms": true/false,
    "contradicts": true/false,
    "confidence": 0.0-1.0,
    "relevant_text": "excerpt from the text that is most relevant",
    "new_facts": [
        {{
            "subject": "subject",
            "relation": "relation",
            "object": "object"
        }}
    ]
}}

The "new_facts" field should contain any new scientific facts found in the text that are related to the subject or object.
Use the following relation types for new facts: {list(self.scientific_relations.keys())}
"""

                response = self.openai_client.chat.completions.create(
                    model=DEFAULT_CONFIG["model_id"],
                    messages=[
                        {"role": "system", "content": "You are a scientific fact-checking assistant. Analyze text and determine if it confirms or contradicts a fact."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                analysis = json.loads(response.choices[0].message.content)
                
                # Update result based on this chunk's analysis
                if analysis.get('confirms', False):
                    result['confirms'] = True
                    if analysis.get('confidence', 0.0) > result['confidence']:
                        result['confidence'] = analysis.get('confidence', 0.0)
                        result['relevant_text'] = analysis.get('relevant_text', "")
                
                if analysis.get('contradicts', False):
                    result['contradicts'] = True
                    if not result['relevant_text'] or analysis.get('confidence', 0.0) > result['confidence']:
                        result['confidence'] = analysis.get('confidence', 0.0)
                        result['relevant_text'] = analysis.get('relevant_text', "")
                
                # Accumulate new facts
                result['new_facts'].extend(analysis.get('new_facts', []))
                
                # If we've found both confirmation and contradiction, we can stop
                if result['confirms'] and result['contradicts']:
                    break
            
            if len(chunks) > max_chunks:
                print(f"  Note: Content was too long. Analyzed first {max_chunks} chunks only.")
            
            return result
            
        except Exception as e:
            print(f"  Error analyzing content: {e}")
            return result
