import cmd
import os
import json
import sqlite3
import pickle
import sys
from typing import Dict, List, Any, Optional
import datetime
import faiss
import numpy as np
import time
import hashlib
import re

# Import the core functions from your existing script
from query_atom_smallworld import run_scientific_reasoning_workflow, SCIENTIFIC_RELATIONS, FIRECRAWL_CONFIG, R1_CONFIG, DEFAULT_CONFIG
from research_verification import FactVerifier
from fact_manager import FactManager  # Import FactManager from fact_manager.py instead
from research_session import SessionManager
from fact_explorer import FactExplorer

class ResearchAssistantCLI(cmd.Cmd):
    """An interactive CLI for scientific research and knowledge management."""
    
    prompt = 'research> '
    intro = """
    =================================================================
    Scientific Research Assistant CLI
    =================================================================
    Type 'help' to see available commands
    Start with 'research <your scientific question>' to begin
    =================================================================
    """
    
    def __init__(self):
        super().__init__()
        # Initialize the session and fact managers
        self.session_manager = SessionManager()
        self.fact_manager = FactManager(self.session_manager)
        self.fact_explorer = FactExplorer(self.fact_manager)
        
        # Add shorthand properties to access session manager's properties
        self.current_session = None
        self.research_data = None
        self.facts = []
        self.deleted_facts = []
        self.databases = None
        self.vector_store = None
        
        # Print existing sessions info
        if self.session_manager.sessions:
            print(f"Found {len(self.session_manager.sessions)} existing research sessions.")
            print("Use 'load <session_id>' to continue a previous session.")
    
    def _load_session(self, session_id):
        """Load a session and update local references."""
        if self.session_manager.load_session(session_id):
            # Update local references after loading
            self.current_session = self.session_manager.current_session
            self.research_data = self.session_manager.research_data
            self.facts = self.session_manager.facts
            self.deleted_facts = self.session_manager.deleted_facts  # Ensure this is updated
            self.databases = self.session_manager.databases
            self.vector_store = self.session_manager.vector_store
            
            # Debug output to verify deleted facts are loaded
            print(f"Loaded {len(self.facts)} active facts and {len(self.deleted_facts)} deleted facts")
            return True
        return False

    def _save_session(self):
        """Save the current session through the session manager."""
        # Update session manager's data with local changes
        self.session_manager.current_session = self.current_session
        self.session_manager.research_data = self.research_data
        self.session_manager.facts = self.facts
        self.session_manager.deleted_facts = self.deleted_facts
        
        return self.session_manager.save_session()
    
    def do_research(self, arg):
        """
        Start a new research process with the given query.
        Usage: research <your scientific question>
        """
        if not arg:
            print("Please provide a scientific question to research.")
            return
            
        print(f"Researching: {arg}")
        print("This may take a while depending on the complexity of the question...")
        
        try:
            # Import the research function
            from query_atom_smallworld import run_scientific_reasoning_workflow, DEFAULT_CONFIG
            
            # Run the research process
            result = run_scientific_reasoning_workflow(arg, agent_config_overrides=DEFAULT_CONFIG)
            
            # Create a new session with the results
            session_id = self.session_manager.create_session(arg, result)
            self._load_session(session_id)
            
            # Print a summary
            print("\nResearch completed.")
            print(f"Created session: {session_id}")
            print(f"Facts discovered: {len(self.facts)}")
            
            # Display the conclusion
            if 'answer' in result.get('results', {}):
                print("\nConclusion:")
                print(result['results']['answer'])
            
        except Exception as e:
            print(f"Error during research: {str(e)}")
    
    def do_load(self, arg):
        """
        Load a research session.
        Usage: load <session_id or number>
        """
        if not self.session_manager.sessions:
            print("No saved sessions found.")
            return
        
        if not arg:
            # Display available sessions
            print("Available sessions:")
            for i, (session_id, metadata) in enumerate(sorted(self.session_manager.sessions.items()), 1):
                print(f"{i}. {session_id}: {metadata['question']} - {metadata['fact_count']} facts - {metadata['timestamp']}")
            
            choice = input("\nEnter session number to load: ")
            try:
                idx = int(choice) - 1
                session_id = list(sorted(self.session_manager.sessions.keys()))[idx]
                self._load_session(session_id)
            except (ValueError, IndexError):
                print("Invalid selection.")
        else:
            # Check if arg is a number
            try:
                # If arg is a number (like "2"), convert to index and get session_id
                idx = int(arg) - 1
                if 0 <= idx < len(self.session_manager.sessions):
                    session_id = list(sorted(self.session_manager.sessions.keys()))[idx]
                    self._load_session(session_id)
                else:
                    print(f"Invalid session number. Please specify a number between 1 and {len(self.session_manager.sessions)}.")
            except ValueError:
                # Not a number, treat as session_id
                self._load_session(arg)
    
    def do_chat(self, arg):
        """
        Start an interactive chat about the research.
        Usage: chat
        """
        if not self.current_session or not self.research_data:
            print("No active research session. Use 'research <question>' or 'load <session_id>' first.")
            return
            
        print("\nStarting chat about your research. Type 'exit' to end the chat.\n")
        
        # Format a system message with research summary
        system_prompt = f"""You are a scientific research assistant chatbot.
You have conducted research on the question: 
"{self.research_data.get('main_question', '')}"

Your conclusion was:
{self.research_data.get('conclusion', 'No conclusion available.')}

You have access to facts from this research and can answer follow-up questions.
Provide accurate, scientific responses based on the research data.
"""
        
        # Prepare messages structure for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ('exit', 'quit', 'q'):
                break
                
            # Save the message to history
            if self.databases:
                cursor = self.databases['relational'].cursor()
                timestamp = datetime.datetime.now().isoformat()
                cursor.execute(
                    'INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)',
                    ('user', user_input, timestamp)
                )
                self.databases['relational'].commit()
            
            # Add to messages list
            messages.append({"role": "user", "content": user_input})
            
            # Retrieve relevant facts for context
            context = self._get_relevant_context(user_input)
            
            # If we have context, add it to the user message
            if context:
                context_message = "Here's some relevant information from our research:\n\n" + context
                messages.append({"role": "user", "content": context_message})
            
            # Generate response using OpenAI API
            try:
                from query_atom_smallworld import DEFAULT_CONFIG
                import openai
                
                client = openai.OpenAI(api_key=DEFAULT_CONFIG["api_key"])
                response = client.chat.completions.create(
                    model="gpt-4.5-preview",  # Use a strong model for chat
                    messages=messages
                )
                
                assistant_response = response.choices[0].message.content
                
                # Display the response
                print(f"Assistant: {assistant_response}")
                
                # Save to history
                if self.databases:
                    cursor = self.databases['relational'].cursor()
                    timestamp = datetime.datetime.now().isoformat()
                    cursor.execute(
                        'INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)',
                        ('assistant', assistant_response, timestamp)
                    )
                    self.databases['relational'].commit()
                
                # Add to messages list
                messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                print(f"Error generating response: {str(e)}")
    
    def _get_relevant_context(self, query, max_facts=10):
        """Get relevant facts and information based on the query."""
        if not self.databases or not self.facts:
            return ""
            
        # Simple keyword matching for now
        keywords = query.lower().split()
        
        # Score facts by keyword matches
        scored_facts = []
        for fact in self.facts:
            score = 0
            # Handle None values by coalescing to empty string before calling lower()
            subject = fact.get('subject', '') or ''
            relation = fact.get('relation', '') or ''
            obj = fact.get('object', '') or ''
            
            combined_text = f"{subject.lower()} {relation.lower()} {obj.lower()}"
            
            for keyword in keywords:
                if keyword in combined_text:
                    score += 1
            
            if score > 0:
                scored_facts.append((score, fact))
        
        # Sort by score (highest first)
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        
        # Format the top facts
        context = ""
        for _, fact in scored_facts[:max_facts]:
            relation = fact.get('relation', 'unknown')
            context += f"- {fact.get('subject', '')} {relation} {fact.get('object', '')}\n"
        
        # Also add some source information if available
        if self.research_data.get('bibliography') and keywords:
            # Look for sources that might have relevant info
            for source in self.research_data.get('bibliography', []):
                if 'chunks' in source:
                    for chunk in source['chunks']:
                        chunk_text = chunk.get('text', '').lower()
                        if any(keyword in chunk_text for keyword in keywords):
                            context += f"\nFrom source {source.get('id', '')}: {chunk.get('text', '')[:300]}...\n"
                            break
        
        return context
    
    def do_facts(self, arg):
        """
        Display the facts extracted from the research.
        Usage: facts [<relation_filter>]
        """
        if not self.facts:
            print("No facts available. Start a research session first.")
            return
            
        if arg:
            # Filter by relation
            filtered_facts = [f for f in self.facts if f.get('relation', '').lower() == arg.lower()]
            print(f"Facts with relation '{arg}':")
        else:
            filtered_facts = self.facts
            print(f"All {len(filtered_facts)} facts:")
        
        # Group facts by relation for better organization
        by_relation = {}
        for fact in filtered_facts:
            relation = fact.get('relation', 'unknown')
            if relation not in by_relation:
                by_relation[relation] = []
            by_relation[relation].append(fact)
        
        # Print facts grouped by relation
        fact_number = 1
        for relation, facts in by_relation.items():
            print(f"\n== {relation} ({len(facts)}) ==")
            for fact in facts:
                relation = fact.get('relation', 'unknown')
                status_text = ""
                
                # Include verification status if available
                if 'verification' in fact:
                    status = fact['verification'].get('status', 'unknown')
                    if status == 'confirmed':
                        status_text = "[CONFIRMED] "
                    elif status == 'unverified':
                        status_text = "[UNVERIFIED] "
                    elif status == 'false':
                        status_text = "[FALSE] "
                    elif status == 'controversial':
                        status_text = "[WARNING] "
                
                print(f"  {fact_number}. {status_text}{fact.get('subject', '')} {relation} {fact.get('object', '')}")
                fact_number += 1
    
    def do_list_facts(self, arg):
        """
        List all facts in the current session.
        Usage: list_facts [limit] [offset]
        """
        # Parse arguments for pagination
        limit = None
        offset = 0
        
        args = arg.split()
        if len(args) >= 1:
            try:
                limit = int(args[0])
            except ValueError:
                print("Invalid limit. Using default.")
        
        if len(args) >= 2:
            try:
                offset = int(args[1])
            except ValueError:
                print("Invalid offset. Using default.")
        
        self.fact_manager.list_facts(limit, offset)
    
    def do_delete_fact(self, arg):
        """
        Delete a fact by its number.
        Usage: delete_fact <fact_number>
        """
        self.fact_manager.delete_fact(arg)
    
    def do_add_fact(self, arg):
        """
        Add a new fact to the current session.
        Usage: add_fact <subject> | <relation> | <object>
        """
        self.fact_manager.add_fact(arg)
    
    def do_facts_by_source(self, arg):
        """
        Display facts grouped by their sources.
        Usage: facts_by_source
        """
        self.fact_manager.facts_by_source()
    
    def do_delete_by_source(self, arg):
        """
        Delete all facts from a specific source.
        Usage: delete_by_source <source_id>
        """
        self.fact_manager.delete_by_source(arg)
    
    def do_list_sources(self, arg):
        """
        List all sources used in the facts.
        Usage: list_sources
        """
        self.fact_manager.list_sources()
    
    def do_fix_sources(self, arg):
        """
        Fix missing source IDs in facts.
        This regenerates proper source identification for facts.
        Usage: fix_sources
        """
        self.fact_manager.fix_sources()
    
    def do_export_json(self, arg):
        """
        Export the current session data to a JSON file using the original format.
        This will ensure compatibility with the query_atom_smallworld.py format.
        Usage: export_json [filename]
        """
        self.fact_manager.export_json(arg)
    
    def do_verify_facts(self, arg):
        """
        Verify facts from a specific source using Firecrawl search.
        Facts will be classified as 'confirmed', 'unverified', 'false', or 'controversial'.
        
        Usage: verify_facts <source_id>
        """
        # Import the verifier
        from research_verification import FactVerifier
        
        # Create a verifier and run the verification
        verifier = FactVerifier(self.fact_manager)
        verifier.verify_facts(self, arg)

    def do_quit(self, arg):
        """Exit the application. Shorthand: q"""
        print("Closing databases and exiting...")
        if self.session_manager:
            # Close any open database connections
            if self.session_manager.databases:
                for db_name, conn in self.session_manager.databases.items():
                    if hasattr(conn, 'close'):
                        try:
                            conn.close()
                        except:
                            pass
                        
        print("Thank you for using the Scientific Research Assistant. Goodbye!")
        return True

    def do_EOF(self, arg):
        # Method with no docstring won't show in help
        print()  # Add newline after ^D
        return self.do_quit(arg)

    def do_list_deleted_facts(self, arg):
        """
        List deleted facts.
        Usage: list_deleted_facts [limit]
        """
        if not self.deleted_facts:  # Use the CLI's direct reference
            print("No deleted facts available.")
            return
        
        limit = None
        if arg:
            try:
                limit = int(arg)
            except ValueError:
                print("Invalid limit. Please provide a number.")
                return
        
        facts_to_display = self.deleted_facts  # Use the CLI's direct reference
        if limit:
            facts_to_display = facts_to_display[:limit]
        
        print(f"Showing {len(facts_to_display)} deleted facts (of {len(self.deleted_facts)} total):")
        for i, fact in enumerate(facts_to_display, 1):
            relation = fact.get('relation', 'unknown')
            print(f"{i}. {fact.get('subject', '')} {relation} {fact.get('object', '')}")
            
            # Show source if available
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
                print(f"   Source: {source_id}")
        
        print("\nUse 'restore_fact <number>' to restore a deleted fact.")

    def do_restore_fact(self, arg):
        """
        Restore a deleted fact back to the active facts list.
        Usage: restore_fact <number>
        """
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(self.deleted_facts):
                fact = self.deleted_facts.pop(idx)
                self.facts.append(fact)
                print(f"Restored fact: {fact.get('subject', '')} {fact.get('relation', '')} {fact.get('object', '')}")
                self.session_manager.save_session()
            else:
                print(f"Invalid fact number. Use 'list_deleted_facts' to see available deleted facts.")
        except ValueError:
            print("Please provide a valid fact number.")

    def do_delete_session(self, arg):
        """
        Delete a session by ID or number.
        Usage: delete_session <session_id or number>
        """
        if not self.session_manager.sessions:
            print("No saved sessions found.")
            return
        
        if not arg:
            # Display available sessions
            print("Available sessions:")
            for i, (session_id, metadata) in enumerate(sorted(self.session_manager.sessions.items()), 1):
                print(f"{i}. {session_id}: {metadata['question']} - {metadata['fact_count']} facts - {metadata['timestamp']}")
            
            choice = input("\nEnter session number to delete: ")
            try:
                idx = int(choice) - 1
                session_id = list(sorted(self.session_manager.sessions.keys()))[idx]
                self.session_manager.delete_session(session_id)
            except (ValueError, IndexError):
                print("Invalid selection.")
        else:
            # Check if arg is a number
            try:
                # If arg is a number (like "2"), convert to index and get session_id
                idx = int(arg) - 1
                if 0 <= idx < len(self.session_manager.sessions):
                    session_id = list(sorted(self.session_manager.sessions.keys()))[idx]
                    self.session_manager.delete_session(session_id)
                else:
                    print(f"Invalid session number. Please specify a number between 1 and {len(self.session_manager.sessions)}.")
            except ValueError:
                # Not a number, treat as session_id
                self.session_manager.delete_session(arg)

    def do_rethink(self, arg):
        """
        Re-analyze the research question with the current set of related facts only.
        Usage: rethink [<optional_additional_question>]
        """
        if not self.current_session or not self.research_data:
            print("No active research session. Start or load a session first.")
            return
        
        question = self.research_data.get('main_question', '')
        if not question:
            print("No research question found in the current session.")
            return
        
        additional_question = arg.strip() if arg else ""
        
        if additional_question:
            print(f"Re-analyzing the research with additional question: {additional_question}")
            full_question = f"{question} Also consider: {additional_question}"
            query = full_question
        else:
            print(f"Re-analyzing the research question: {question}")
            full_question = question
            query = question
        
        print(f"\nStarting search for facts related to: {query}")
        
        # Get related facts using fact_explorer
        try:
            if not hasattr(self, '_openai_client'):
                import openai
                self._openai_client = openai.OpenAI(
                    api_key=R1_CONFIG["api_key"],
                    base_url=R1_CONFIG["endpoint"]
                )
            
            # Prompt to analyze the query for better fact retrieval
            prompt = f"""Analyze this query and extract key terms that would be most relevant for finding related scientific facts.
Differentiate between important terms and non-important terms.
Return the response as a JSON object with two lists: "key_terms" and "context_terms"

Query: {query}

Example response format:
{{
    "key_terms": ["most", "important", "terms"],
    "context_terms": ["less", "important", "context", "terms"]
}}"""

            print("Analyzing query terms...")
            response = self._openai_client.chat.completions.create(
                model=R1_CONFIG["model_id"],
                messages=[
                    {"role": "system", "content": "You are a scientific term analyzer. Extract and classify terms from queries."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            filtered_facts = []
            
            try:
                # Extract JSON from markdown code blocks if present
                content = response.choices[0].message.content
                # Look for JSON content, with or without markdown formatting
                if "```" in content:
                    # Extract content between markdown code blocks if present
                    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                    else:
                        json_content = content
                else:
                    json_content = content
                
                # Parse the response and handle potential JSON errors
                terms = json.loads(json_content)
                key_terms = terms.get("key_terms", [])
                context_terms = terms.get("context_terms", [])
                
                print(f"\nKey search terms: {', '.join(key_terms)}")
                if context_terms:
                    print(f"Context terms: {', '.join(context_terms)}")
                print("\nSearching for related facts...")
                
                # First try with key terms
                filtered_facts = self.fact_explorer.filter_facts(" ".join(key_terms))
                
                # If not enough results, include context terms
                if len(filtered_facts) < 5 and context_terms:
                    print("\nExpanding search with context terms...")
                    context_filtered = self.fact_explorer.filter_facts(" ".join(context_terms))
                    # Combine results, removing duplicates
                    seen = {(f[0].get('subject', ''), f[0].get('relation', ''), f[0].get('object', '')) 
                           for f in filtered_facts}
                    for fact_score in context_filtered:
                        fact_tuple = (fact_score[0].get('subject', ''), 
                                    fact_score[0].get('relation', ''), 
                                    fact_score[0].get('object', ''))
                        if fact_tuple not in seen:
                            filtered_facts.append(fact_score)
                            seen.add(fact_tuple)
                
            except json.JSONDecodeError as je:
                print(f"Error parsing OpenAI response. Falling back to direct search.")
                print(f"Using query directly: {query}")
                filtered_facts = self.fact_explorer.filter_facts(query)
                
            except Exception as e:
                print(f"Error analyzing query: {str(e)}. Falling back to direct search.")
                filtered_facts = self.fact_explorer.filter_facts(query)
                
            # Prepare context with ONLY related facts
            if not filtered_facts:
                print("\nNo related facts found. Cannot re-analyze the question.")
                return
                
            print(f"\nFound {len(filtered_facts)} related facts for analysis.")
            
            # Extract just the facts from the scored fact tuples
            related_facts = [fact_score[0] for fact_score in filtered_facts]
            
            # Build the facts context string
            facts_context = ""
            for i, fact in enumerate(related_facts, 1):
                relation = fact.get('relation', 'unknown')
                facts_context += f"{i}. {fact.get('subject', '')} {relation} {fact.get('object', '')}\n"
        
            # Generate new analysis with only related facts
            prompt = f"""Based on the following RELATED facts, please re-analyze this research question:
Question: {full_question}

Related Facts:
{facts_context}

Previous conclusion:
{self.research_data.get('conclusion', 'No previous conclusion.')}

Please provide:
1. A comprehensive analysis
2. A revised conclusion
3. Any new insights or hypotheses
"""
            
            print("\nGenerating new analysis with related facts only. This may take a moment...\n")
            
            response = self._openai_client.chat.completions.create(
                model=R1_CONFIG["model_id"],
                messages=[
                    {"role": "system", "content": "You are a scientific research assistant. Analyze the facts provided and answer the research question thoroughly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            new_analysis = response.choices[0].message.content
            
            # Update the research data
            self.research_data['revised_analysis'] = new_analysis
            self.research_data['last_revised'] = datetime.datetime.now().isoformat()
            self.research_data['used_related_facts_only'] = True
            self.research_data['related_facts_count'] = len(related_facts)
            
            # Save the session
            self._save_session()
            
            # Print the results
            print("\n=== New Analysis (Based on Related Facts Only) ===\n")
            print(new_analysis)
            
        except Exception as e:
            print(f"Error during re-analysis: {str(e)}")

    def do_related_facts(self, arg):
        """
        Find facts related to a given query by matching terms in subject or object positions.
        Usage: related_facts <query>
        Example: related_facts quantum computing
                related_facts "neural networks"
        """
        if not self.facts:
            print("No facts available. Start a research session first.")
            return
        
        if not arg:
            if not self.research_data or 'main_question' not in self.research_data:
                print("Please provide a search query.")
                print("Usage: related_facts <query>")
                return
            arg = self.research_data['main_question']
            print(f"Using original research question as query: {arg}")

        print(f"\nStarting search with query: {arg}")

        # First analyze the query to extract key terms
        try:
            if not hasattr(self, '_openai_client'):
                import openai
                self._openai_client = openai.OpenAI(
                    api_key=R1_CONFIG["api_key"],
                    base_url=R1_CONFIG["endpoint"]
                )
            
            # Prompt to analyze the query
            prompt = f"""Analyze this query and extract key terms that would be most relevant for finding related scientific facts.
Differentiate between important terms and non-important terms.
Return the response as a JSON object with two lists: "key_terms" and "context_terms"

Query: {arg}

Example response format:
{{
    "key_terms": ["most", "important", "terms"],
    "context_terms": ["less", "important", "context", "terms"]
}}"""

            print("Analyzing query terms...")
            response = self._openai_client.chat.completions.create(
                model=R1_CONFIG["model_id"],
                messages=[
                    {"role": "system", "content": "You are a scientific term analyzer. Extract and classify terms from queries."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            print("\nDebug - Raw OpenAI response content:")
            print(response.choices[0].message.content)
            print("\nAttempting to parse response...")
            
            try:
                # Extract JSON from markdown code blocks if present
                content = response.choices[0].message.content
                # Look for JSON content, with or without markdown formatting
                if "```" in content:
                    # Extract content between markdown code blocks if present
                    # Match anything between code blocks, non-greedy
                    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                        print("Extracted JSON from code block")
                    else:
                        # Fallback to original content
                        json_content = content
                else:
                    json_content = content
                
                # Parse the response and handle potential JSON errors
                terms = json.loads(json_content)
                key_terms = terms.get("key_terms", [])
                context_terms = terms.get("context_terms", [])
                
                print(f"\nKey search terms: {', '.join(key_terms)}")
                if context_terms:
                    print(f"Context terms: {', '.join(context_terms)}")
                print("\nSearching for related facts...")
                
                # First try with key terms
                filtered_facts = self.fact_explorer.filter_facts(" ".join(key_terms))
                
                # If not enough results, include context terms
                if len(filtered_facts) < 5 and context_terms:
                    print("\nExpanding search with context terms...")
                    context_filtered = self.fact_explorer.filter_facts(" ".join(context_terms))
                    # Combine results, removing duplicates
                    seen = {(f[0].get('subject', ''), f[0].get('relation', ''), f[0].get('object', '')) 
                           for f in filtered_facts}
                    for fact_score in context_filtered:
                        fact_tuple = (fact_score[0].get('subject', ''), 
                                    fact_score[0].get('relation', ''), 
                                    fact_score[0].get('object', ''))
                        if fact_tuple not in seen:
                            filtered_facts.append(fact_score)
                            seen.add(fact_tuple)
                
                # Print results
                if filtered_facts:
                    print(f"\nFound {len(filtered_facts)} related facts:")
                    self.fact_explorer.print_filtered_facts(filtered_facts)
                else:
                    print(f"\nNo related facts found for the given terms.")
                    
            except json.JSONDecodeError as je:
                print(f"Error parsing OpenAI response. Falling back to direct search.")
                print(f"Using query directly: {arg}")
                filtered_facts = self.fact_explorer.filter_facts(arg)
                if filtered_facts:
                    self.fact_explorer.print_filtered_facts(filtered_facts)
                else:
                    print("\nNo related facts found for query: " + arg)
            
        except Exception as e:
            print(f"Error analyzing query: {str(e)}")
            # Fall back to direct search without term analysis
            filtered_facts = self.fact_explorer.filter_facts(arg)
            if filtered_facts:
                self.fact_explorer.print_filtered_facts(filtered_facts)
            else:
                print("\nNo related facts found for query: " + arg)

if __name__ == "__main__":
    # Import necessary variables/functions from query_atom_smallworld
    try:
        from query_atom_smallworld import DEFAULT_CONFIG, SCIENTIFIC_RELATIONS
    except ImportError:
        print("Error: Could not import query_atom_smallworld.py")
        print("Make sure it's in the same directory as this script.")
        sys.exit(1)
        
    print("\nInitializing Scientific Research Assistant...")
    ResearchAssistantCLI().cmdloop()
