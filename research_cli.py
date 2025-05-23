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
import uuid

# Import the core functions from your existing script
from query_atom_smallworld import run_scientific_reasoning_workflow, SCIENTIFIC_RELATIONS, FIRECRAWL_CONFIG, R1_CONFIG, DEFAULT_CONFIG
from research_verification import FactVerifier
from fact_manager import FactManager  # Import FactManager from fact_manager.py instead
from research_session import SessionManager
from fact_explorer import FactExplorer
from thought_chain import ThoughtChain, ThoughtNode, extract_steps_from_text, thoughts_text_to_nodes, build_thought_chain

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
           research -r SESSION_ID <your scientific question>  (to resume from a session)
           research -r NUMBER <your scientific question>      (to resume by session number)
        """
        if not arg:
            print("Please provide a scientific question to research.")
            return
        
        # Check if this is a resume request
        resume_session = False
        session_id = None
        
        # Parse for resume flag
        if arg.startswith('-r '):
            parts = arg[3:].strip().split(' ', 1)
            if len(parts) < 2:
                print("Error: When using -r flag, specify both SESSION_ID/NUMBER and your question.")
                print("Usage: research -r SESSION_ID your scientific question")
                print("       research -r 2 your scientific question (to use session number)")
                return
            
            session_identifier = parts[0]
            scientific_question = parts[1]
            resume_session = True
            
            # Check if it's a number
            try:
                # If it's a number (like "2"), convert to session_id
                idx = int(session_identifier) - 1
                if 0 <= idx < len(self.session_manager.sessions):
                    session_id = list(sorted(self.session_manager.sessions.keys()))[idx]
                else:
                    print(f"Invalid session number. Please specify a number between 1 and {len(self.session_manager.sessions)}.")
                    # Show available sessions
                    print("Available sessions:")
                    for i, (sid, metadata) in enumerate(sorted(self.session_manager.sessions.items()), 1):
                        print(f"{i}. {sid}: {metadata['question']} - {metadata['fact_count']} facts - {metadata['timestamp']}")
                    return
            except ValueError:
                # Not a number, use as session_id
                session_id = session_identifier
            
            # Verify the session exists
            if session_id not in self.session_manager.sessions:
                print(f"Error: Session {session_id} not found.")
                available_sessions = list(self.session_manager.sessions.keys())
                if available_sessions:
                    print("Available sessions:")
                    for i, sid in enumerate(available_sessions, 1):
                        print(f"{i}. {sid}: {self.session_manager.sessions[sid].get('question', 'Unknown')}")
                return
            
            print(f"Resuming research on '{scientific_question}' from session {session_id}")
            # Load the session to resume
            self._load_session(session_id)
        else:
            scientific_question = arg
            print(f"Researching: {scientific_question}")
            
            # Create an initial session immediately to enable incremental saving
            session_id = self.session_manager.create_session(scientific_question, {"main_question": scientific_question, "results": {}})
            self._load_session(session_id)
            print(f"Created session: {session_id}")
        
        print("This may take a while depending on the complexity of the question...")
        
        try:
            # Import the research function
            from query_atom_smallworld import run_scientific_reasoning_workflow, DEFAULT_CONFIG
            
            # Run the research process with the session_id for incremental saving
            result = run_scientific_reasoning_workflow(
                scientific_question, 
                agent_config_overrides=DEFAULT_CONFIG,
                session_id=session_id,
                session_manager=self.session_manager
            )
            
            # The session has already been updated incrementally, so we just load the final state
            self._load_session(session_id)
            
            # Print a summary
            print("\nResearch completed.")
            print(f"Session: {session_id}")
            print(f"Facts discovered: {len(self.facts)}")
            
            # Display the conclusion
            if 'conclusions' in result:
                print("\nConclusion:")
                print(result['conclusions'])
            elif 'answer' in result.get('results', {}):
                print("\nConclusion:")
                print(result['results']['answer'])
            
        except Exception as e:
            print(f"Error during research: {str(e)}")
            print(f"Session {session_id} has been saved. You can resume later using 'resume' or 'research -r {session_id} {scientific_question}'.")
    
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
            
            print("Searching for relevant research context...")
            # Retrieve relevant facts for context
            context = self._get_relevant_context(user_input)
            
            # If we have context, add it to the user message
            if context:
                context_message = "Here's some relevant information from our research:\n\n" + context
                messages.append({"role": "user", "content": context_message})
            
            # Generate response using OpenAI API
            try:
                print("Generating response...")
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
                
                print("Saving chat history...")
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
                    {"role": "user", "content": prompt}
                ]
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

    def do_resume(self, arg):
        """
        Resume research on the current or specified session.
        Usage: resume [session_number]
               resume              (to resume current session)
               resume 3            (to resume session number 3)
        """
        # Make sure we have sessions
        if not self.session_manager.sessions:
            print("No saved sessions found.")
            return
        
        # Determine which session to use
        session_id = None
        
        if not arg:
            # Use current session if loaded
            if self.current_session:
                session_id = self.current_session
            else:
                print("No active session. Please specify a session number or load a session first.")
                print("Available sessions:")
                for i, (sid, metadata) in enumerate(sorted(self.session_manager.sessions.items()), 1):
                    print(f"{i}. {sid}: {metadata['question']} - {metadata['fact_count']} facts - {metadata['timestamp']}")
                return
        else:
            # Try to parse as session number
            try:
                idx = int(arg) - 1
                if 0 <= idx < len(self.session_manager.sessions):
                    session_id = list(sorted(self.session_manager.sessions.keys()))[idx]
                else:
                    print(f"Invalid session number. Please specify a number between 1 and {len(self.session_manager.sessions)}.")
                    return
            except ValueError:
                print("Please provide a valid session number.")
                return
        
        # Load the session if not already loaded
        if self.current_session != session_id:
            if not self._load_session(session_id):
                print(f"Failed to load session {session_id}")
                return
        
        # Get the question from the loaded session
        scientific_question = self.research_data.get('main_question', '')
        if not scientific_question:
            print("Could not find the research question in the session data.")
            return
        
        print(f"Resuming research on: {scientific_question}")
        print("This may take a while depending on the complexity of the question...")
        
        try:
            # Import the research function
            from query_atom_smallworld import run_scientific_reasoning_workflow, DEFAULT_CONFIG
            
            # Run the research process with the session_id for incremental saving
            result = run_scientific_reasoning_workflow(
                scientific_question, 
                agent_config_overrides=DEFAULT_CONFIG,
                session_id=session_id,
                session_manager=self.session_manager
            )
            
            # The session has already been updated incrementally, so we just load the final state
            self._load_session(session_id)
            
            # Print a summary
            print("\nResearch completed.")
            print(f"Session: {session_id}")
            print(f"Facts discovered: {len(self.facts)}")
            
            # Display the conclusion
            if 'conclusions' in result:
                print("\nConclusion:")
                print(result['conclusions'])
            elif 'answer' in result.get('results', {}):
                print("\nConclusion:")
                print(result['results']['answer'])
            
        except Exception as e:
            print(f"Error during research: {str(e)}")
            print(f"Session {session_id} has been saved. You can resume later using 'resume' command.")

    def do_convert_thoughts(self, arg):
        """
        Convert thoughts from research_data['thoughts'] (separated by double newlines) to ThoughtNode dicts and store in research_data['thought_nodes'].
        Also creates a ThoughtChain and sets it as the current thought chain.
        Usage: convert_thoughts
        """
        if 'thoughts' not in self.research_data or not self.research_data['thoughts']:
            print("No thoughts found in research_data. Run research first to generate thoughts.")
            return
            
        thoughts_text = self.research_data['thoughts']
        nodes = thoughts_text_to_nodes(thoughts_text)
        node_dicts = [node.to_dict() for node in nodes]
        if 'thought_nodes' not in self.research_data:
            self.research_data['thought_nodes'] = []
        self.research_data['thought_nodes'].extend(node_dicts)
        
        # Create ThoughtChain and set as current
        chain = build_thought_chain(nodes)
        self.research_data['current_thought_chain'] = {
            'root_id': chain.root_id
        }
        print(f"Added {len(node_dicts)} thought nodes to research_data['thought_nodes'].")
        print(f"Created ThoughtChain with root_id: {chain.root_id}")
        
        # Save the session
        self._save_session()
        print("Session saved.")

    def do_add_thoughtnode(self, arg):
        """
        Interactively add a single ThoughtNode to research_data['thought_nodes'].
        Usage: add_thoughtnode
        """
        # Ensure the thought_nodes list exists
        if 'thought_nodes' not in self.research_data:
            self.research_data['thought_nodes'] = []
        nodes = self.research_data['thought_nodes']

        # List existing nodes
        if nodes:
            print("Existing ThoughtNodes:")
            for i, node in enumerate(nodes, 1):
                print(f"  {i}. {node['text'][:60]}{'...' if len(node['text']) > 60 else ''}")
        else:
            print("No existing ThoughtNodes. The new node will be a root node.")

        # Prompt for parent node
        parent_id = None
        if nodes:
            parent_input = input("Enter parent node number (or 'none' for root): ").strip()
            if parent_input.lower() != 'none':
                try:
                    parent_index = int(parent_input) - 1
                    if 0 <= parent_index < len(nodes):
                        parent_id = nodes[parent_index]['id']
                    else:
                        print("Invalid parent number. Aborting.")
                        return
                except ValueError:
                    print("Invalid parent number. Aborting.")
                    return

        # Prompt for node text
        text = arg.strip() if arg else input("Enter text for the new thought node: ").strip()
        if not text:
            print("No text provided. Aborting.")
            return

        node_id = str(uuid.uuid4())
        node = ThoughtNode(
            id=node_id,
            text=text,
            parent_id=parent_id,
            children_ids=[],
            metadata={}
        )
        # Add to parent's children_ids if applicable
        if parent_id:
            for n in nodes:
                if n['id'] == parent_id:
                    # If parent already has children, remove them from the chain to maintain linearity
                    existing_children = n.get('children_ids', [])
                    if existing_children:
                        print(f"Truncating chain: removing {len(existing_children)} existing children from chain structure.")
                        # Remove existing children from chain (but keep in storage)
                        for child_id in existing_children:
                            for child_node in nodes:
                                if child_node['id'] == child_id:
                                    child_node['parent_id'] = None
                                    child_node['children_ids'] = []
                                    break
                    
                    # Set new node as the only child
                    n['children_ids'] = [node_id]
                    break
        # Add new node
        self.research_data['thought_nodes'].append(node.to_dict())
        print(f"Added ThoughtNode (parent: {parent_input if parent_input.lower() != 'none' else 'none'})")
        
        # Save the session
        self._save_session()

    def do_edit_thoughtnode(self, arg):
        """
        Interactively edit the text of a ThoughtNode using numbered selection.
        Usage: edit_thoughtnode
        """
        if 'thought_nodes' not in self.research_data or not self.research_data['thought_nodes']:
            print("No thought nodes to edit.")
            return
        nodes = self.research_data['thought_nodes']
        print("Available ThoughtNodes:")
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node['text'][:80]}{'...' if len(node['text']) > 80 else ''}")
        
        try:
            choice = input("Enter the number of the node to edit: ").strip()
            node_index = int(choice) - 1
            
            if 0 <= node_index < len(nodes):
                node = nodes[node_index]
                print(f"\nCurrent text: {node['text']}")
                new_text = input("Enter new text: ").strip()
                if not new_text:
                    print("No new text provided. Aborting.")
                    return
                
                # Truncate chain after this node since we're changing the reasoning
                existing_children = node.get('children_ids', [])
                if existing_children:
                    print(f"Truncating chain: removing {len(existing_children)} subsequent nodes from chain structure.")
                    # Remove all children from chain (but keep in storage)
                    for child_id in existing_children:
                        for child_node in nodes:
                            if child_node['id'] == child_id:
                                child_node['parent_id'] = None
                                child_node['children_ids'] = []
                    # Clear this node's children
                    node['children_ids'] = []
                
                node['text'] = new_text
                print(f"Updated ThoughtNode {choice}.")
                
                # Save the session
                self._save_session()
            else:
                print("Invalid number. Please select a valid node number.")
        except ValueError:
            print("Please enter a valid number.")

    def do_view_thoughtchain(self, arg):
        """
        Display the current thought chain as an ASCII art flowchart with full text and numbered nodes.
        Usage: view_thoughtchain
        """
        if 'current_thought_chain' not in self.research_data or not self.research_data['current_thought_chain']:
            print("No current thought chain set.")
            return
        
        root_id = self.research_data['current_thought_chain'].get('root_id')
        if not root_id:
            print("No root_id in current thought chain.")
            return
            
        if 'thought_nodes' not in self.research_data or not self.research_data['thought_nodes']:
            print("No thought nodes available.")
            return
            
        # Create a lookup dict for nodes and number mapping
        nodes_dict = {node['id']: node for node in self.research_data['thought_nodes']}
        
        # Create number mapping (1-based indexing)
        id_to_number = {}
        for i, node in enumerate(self.research_data['thought_nodes']):
            id_to_number[node['id']] = i + 1
        
        if root_id not in nodes_dict:
            print(f"Root node {root_id} not found in thought nodes.")
            return
            
        print("Current ThoughtChain Flowchart:")
        print("=" * 120)
        self._display_flowchart(root_id, nodes_dict, id_to_number)
    
    def _display_flowchart(self, root_id, nodes_dict, id_to_number):
        """Display the thought chain as ASCII art flowchart."""
        # Get the layout of all nodes
        layout = self._calculate_layout(root_id, nodes_dict)
        
        # Render the flowchart
        self._render_flowchart(layout, nodes_dict, id_to_number)
    
    def _calculate_layout(self, node_id, nodes_dict, x=0, y=0, visited=None):
        """Calculate positions for all nodes in the flowchart."""
        if visited is None:
            visited = set()
        
        if node_id in visited or node_id not in nodes_dict:
            return {}
        
        visited.add(node_id)
        node = nodes_dict[node_id]
        
        layout = {node_id: {'x': x, 'y': y, 'node': node}}
        
        children = node.get('children_ids', [])
        if children:
            # Calculate spacing for children based on text width - increase spacing
            child_spacing = max(6, len(children) * 3)
            start_x = x - (len(children) - 1) * child_spacing // 2
            
            for i, child_id in enumerate(children):
                child_x = start_x + i * child_spacing
                child_layout = self._calculate_layout(child_id, nodes_dict, child_x, y + 2, visited)
                layout.update(child_layout)
        
        return layout
    
    def _render_flowchart(self, layout, nodes_dict, id_to_number):
        """Render the flowchart as ASCII art."""
        if not layout:
            return
        
        # Find bounds
        min_x = min(pos['x'] for pos in layout.values())
        max_x = max(pos['x'] for pos in layout.values())
        min_y = min(pos['y'] for pos in layout.values())
        max_y = max(pos['y'] for pos in layout.values())
        
        # Adjust coordinates to be positive
        for pos in layout.values():
            pos['x'] -= min_x
            pos['y'] -= min_y
        
        # Calculate max text width for proper grid sizing
        max_text_width = 0
        for node_id, pos in layout.items():
            text = pos['node']['text']
            max_text_width = max(max_text_width, len(text))
        
        # Ensure minimum width for readability
        box_width = max(30, min(80, max_text_width + 4))
        
        # Create grid with dynamic sizing - increase spacing
        width = (max_x - min_x + 1) * (box_width + 15) + 30
        height = (max_y - min_y + 1) * 10 + 15
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Draw nodes and connections
        for node_id, pos in layout.items():
            self._draw_node(grid, pos, node_id, id_to_number, box_width)
            self._draw_connections(grid, pos, nodes_dict, layout, box_width)
        
        # Print the grid
        for row in grid:
            print(''.join(row).rstrip())
    
    def _draw_node(self, grid, pos, node_id, id_to_number, box_width):
        """Draw a single node box with full text."""
        node = pos['node']
        text = node['text']
        node_number = id_to_number.get(node_id, '?')
        
        # Wrap text to fit in box but show full text
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= box_width - 2:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Calculate box position and size
        box_height = len(lines) + 4  # Extra space for borders and number
        start_x = pos['x'] * (box_width + 15)
        start_y = pos['y'] * 10
        
        # Draw box
        try:
            # Top border
            for i in range(box_width):
                if start_y < len(grid) and start_x + i < len(grid[0]):
                    grid[start_y][start_x + i] = '─' if i > 0 and i < box_width - 1 else ('┌' if i == 0 else '┐')
            
            # Node number line
            number_text = f"Node {node_number}"
            if start_y + 1 < len(grid):
                if start_x < len(grid[0]):
                    grid[start_y + 1][start_x] = '│'
                if start_x + box_width - 1 < len(grid[0]):
                    grid[start_y + 1][start_x + box_width - 1] = '│'
                
                # Center the number
                padding = (box_width - 2 - len(number_text)) // 2
                for j, char in enumerate(number_text):
                    if start_x + 1 + padding + j < len(grid[0]):
                        grid[start_y + 1][start_x + 1 + padding + j] = char
            
            # Separator line
            if start_y + 2 < len(grid):
                for i in range(box_width):
                    if start_x + i < len(grid[0]):
                        grid[start_y + 2][start_x + i] = '─' if i > 0 and i < box_width - 1 else ('├' if i == 0 else '┤')
            
            # Text content
            for i, line in enumerate(lines):
                row_y = start_y + 3 + i
                if row_y < len(grid):
                    if start_x < len(grid[0]):
                        grid[row_y][start_x] = '│'
                    if start_x + box_width - 1 < len(grid[0]):
                        grid[row_y][start_x + box_width - 1] = '│'
                    
                    # Add text
                    for j, char in enumerate(line):
                        if start_x + 1 + j < len(grid[0]):
                            grid[row_y][start_x + 1 + j] = char
            
            # Bottom border
            bottom_y = start_y + 3 + len(lines)
            if bottom_y < len(grid):
                for i in range(box_width):
                    if start_x + i < len(grid[0]):
                        grid[bottom_y][start_x + i] = '─' if i > 0 and i < box_width - 1 else ('└' if i == 0 else '┘')
        
        except IndexError:
            pass  # Skip if out of bounds
    
    def _draw_connections(self, grid, pos, nodes_dict, layout, box_width):
        """Draw connections from this node to its children."""
        node = pos['node']
        children = node.get('children_ids', [])
        
        if not children:
            return
        
        # Calculate connection points
        parent_x = pos['x'] * (box_width + 15) + box_width // 2
        text_lines = len(node['text'].split('\n')) if '\n' in node['text'] else len([line for line in [node['text'][i:i+box_width-2] for i in range(0, len(node['text']), box_width-2)] if line])
        parent_y = pos['y'] * 10 + 4 + max(1, text_lines) + 2
        
        try:
            # Draw vertical line down from parent
            if parent_y < len(grid) and parent_x < len(grid[0]):
                grid[parent_y][parent_x] = '│'
            
            if len(children) == 1:
                # Single child - straight line
                child_pos = layout.get(children[0])
                if child_pos:
                    child_x = child_pos['x'] * (box_width + 15) + box_width // 2
                    child_y = child_pos['y'] * 10
                    
                    # Draw vertical line to child
                    for y in range(parent_y + 1, child_y):
                        if y < len(grid) and parent_x < len(grid[0]):
                            grid[y][parent_x] = '│'
                    
                    # Draw arrow to child
                    if child_y - 1 < len(grid) and child_x < len(grid[0]):
                        grid[child_y - 1][child_x] = '▼'
            
            elif len(children) > 1:
                # Multiple children - horizontal distribution
                child_positions = []
                for child_id in children:
                    if child_id in layout:
                        child_pos = layout[child_id]
                        child_x = child_pos['x'] * (box_width + 15) + box_width // 2
                        child_y = child_pos['y'] * 10
                        child_positions.append((child_x, child_y))
                
                if child_positions:
                    # Draw horizontal line
                    min_child_x = min(x for x, y in child_positions)
                    max_child_x = max(x for x, y in child_positions)
                    horizontal_y = parent_y + 1
                    
                    if horizontal_y < len(grid):
                        for x in range(min_child_x, max_child_x + 1):
                            if x < len(grid[0]):
                                grid[horizontal_y][x] = '─'
                        
                        # Draw vertical lines to each child
                        for child_x, child_y in child_positions:
                            # Connect to horizontal line
                            if parent_x < len(grid[0]):
                                grid[horizontal_y][parent_x] = '┬'
                            if child_x < len(grid[0]):
                                grid[horizontal_y][child_x] = '┬'
                            
                            # Draw down to child
                            for y in range(horizontal_y + 1, child_y):
                                if y < len(grid) and child_x < len(grid[0]):
                                    grid[y][child_x] = '│'
                            
                            # Draw arrow
                            if child_y - 1 < len(grid) and child_x < len(grid[0]):
                                grid[child_y - 1][child_x] = '▼'
        
        except IndexError:
            pass  # Skip if out of bounds

    def do_move_thoughtnode(self, arg):
        """
        Move a thought node (and its subtree) to a new parent in the current thought chain.
        Usage: move_thoughtnode
        """
        if 'current_thought_chain' not in self.research_data or not self.research_data['current_thought_chain']:
            print("No current thought chain set.")
            return
            
        if 'thought_nodes' not in self.research_data or not self.research_data['thought_nodes']:
            print("No thought nodes available.")
            return
            
        nodes = self.research_data['thought_nodes']
        nodes_dict = {node['id']: node for node in nodes}
        
        print("Current ThoughtNodes:")
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node['text'][:50]}{'...' if len(node['text']) > 50 else ''}")
        
        try:
            choice = input("Enter the number of the node to move: ").strip()
            node_index = int(choice) - 1
            
            if not (0 <= node_index < len(nodes)):
                print("Invalid node number.")
                return
                
            node_id = nodes[node_index]['id']
            
            print("Available parent nodes (or 'none' for root):")
            for i, node in enumerate(nodes, 1):
                if i != int(choice):  # Can't be parent of itself
                    print(f"  {i}. {node['text'][:50]}{'...' if len(node['text']) > 50 else ''}")
            
            parent_choice = input("Enter new parent number (or 'none' for root): ").strip()
            if parent_choice.lower() == 'none':
                new_parent_id = None
            else:
                try:
                    parent_index = int(parent_choice) - 1
                    if 0 <= parent_index < len(nodes) and parent_index != node_index:
                        new_parent_id = nodes[parent_index]['id']
                    else:
                        print("Invalid parent number.")
                        return
                except ValueError:
                    print("Invalid parent number.")
                    return
                    
            # Update the node's parent
            node = nodes_dict[node_id]
            old_parent_id = node.get('parent_id')
            
            # Remove from old parent's children
            if old_parent_id and old_parent_id in nodes_dict:
                old_parent = nodes_dict[old_parent_id]
                if node_id in old_parent.get('children_ids', []):
                    old_parent['children_ids'].remove(node_id)
            
            # Set new parent
            node['parent_id'] = new_parent_id
            
            # Add to new parent's children
            if new_parent_id and new_parent_id in nodes_dict:
                new_parent = nodes_dict[new_parent_id]
                if 'children_ids' not in new_parent:
                    new_parent['children_ids'] = []
                if node_id not in new_parent['children_ids']:
                    new_parent['children_ids'].append(node_id)
            
            print(f"Moved node {choice} to parent {parent_choice}")
            
            # Save the session
            self._save_session()
            
        except ValueError:
            print("Please enter a valid number.")

    def do_remove_thoughtnode(self, arg):
        """
        Remove a thought node from the current thought chain structure (but keep in thought_nodes).
        Usage: remove_thoughtnode
        """
        if 'current_thought_chain' not in self.research_data or not self.research_data['current_thought_chain']:
            print("No current thought chain set.")
            return
            
        if 'thought_nodes' not in self.research_data or not self.research_data['thought_nodes']:
            print("No thought nodes available.")
            return
            
        nodes = self.research_data['thought_nodes']
        nodes_dict = {node['id']: node for node in nodes}
        
        print("Current ThoughtNodes:")
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node['text'][:50]}{'...' if len(node['text']) > 50 else ''}")
        
        try:
            choice = input("Enter the number of the node to remove from chain: ").strip()
            node_index = int(choice) - 1
            
            if not (0 <= node_index < len(nodes)):
                print("Invalid node number.")
                return
                
            node_id = nodes[node_index]['id']
            node = nodes_dict[node_id]
            children_ids = node.get('children_ids', [])
            
            # Handle children if any
            if children_ids:
                print(f"This node has {len(children_ids)} children. What should happen to them?")
                print("1. Remove them from chain too")
                print("2. Re-parent them to this node's parent")
                child_choice = input("Enter choice (1 or 2): ").strip()
                
                if child_choice == "1":
                    # Remove children from chain recursively
                    for child_id in children_ids:
                        if child_id in nodes_dict:
                            nodes_dict[child_id]['parent_id'] = None
                            nodes_dict[child_id]['children_ids'] = []
                elif child_choice == "2":
                    # Re-parent children to this node's parent
                    new_parent_id = node.get('parent_id')
                    for child_id in children_ids:
                        if child_id in nodes_dict:
                            nodes_dict[child_id]['parent_id'] = new_parent_id
                            # Add to new parent's children if new parent exists
                            if new_parent_id and new_parent_id in nodes_dict:
                                new_parent = nodes_dict[new_parent_id]
                                if 'children_ids' not in new_parent:
                                    new_parent['children_ids'] = []
                                if child_id not in new_parent['children_ids']:
                                    new_parent['children_ids'].append(child_id)
                else:
                    print("Invalid choice. Aborting.")
                    return
            
            # Remove from parent's children
            parent_id = node.get('parent_id')
            if parent_id and parent_id in nodes_dict:
                parent = nodes_dict[parent_id]
                if node_id in parent.get('children_ids', []):
                    parent['children_ids'].remove(node_id)
            
            # Clear this node's relationships but keep the node
            node['parent_id'] = None
            node['children_ids'] = []
            
            print(f"Removed node {choice} from thought chain structure.")
            
            # Save the session
            self._save_session()
            
        except ValueError:
            print("Please enter a valid number.")

    def do_set_current_chain(self, arg):
        """
        Set which node should be the root of the current thought chain.
        Usage: set_current_chain
        """
        if 'thought_nodes' not in self.research_data or not self.research_data['thought_nodes']:
            print("No thought nodes available.")
            return
            
        nodes = self.research_data['thought_nodes']
        
        # Find potential root nodes (nodes with no parent)
        root_candidates = [node for node in nodes if node.get('parent_id') is None]
        
        if not root_candidates:
            print("No potential root nodes found (all nodes have parents).")
            return
            
        print("Available root node candidates:")
        for i, node in enumerate(root_candidates, 1):
            print(f"  {i}. {node['text'][:50]}{'...' if len(node['text']) > 50 else ''}")
        
        try:
            choice = input("Enter the number of the node to set as current chain root: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(root_candidates):
                selected_node = root_candidates[idx]
                if 'current_thought_chain' not in self.research_data:
                    self.research_data['current_thought_chain'] = {}
                self.research_data['current_thought_chain']['root_id'] = selected_node['id']
                print(f"Set current thought chain root to node {choice}.")
                
                # Save the session
                self._save_session()
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")

    def do_continue_reasoning(self, arg):
        """
        Convert the current thought chain to text and send it to deepseek-V3 for continued reasoning.
        Usage: continue_reasoning [additional_context]
        """
        if 'current_thought_chain' not in self.research_data or not self.research_data['current_thought_chain']:
            print("No current thought chain set. Use 'set_current_chain' first.")
            return
            
        root_id = self.research_data['current_thought_chain'].get('root_id')
        if not root_id:
            print("No root_id in current thought chain.")
            return
            
        if 'thought_nodes' not in self.research_data or not self.research_data['thought_nodes']:
            print("No thought nodes available.")
            return
            
        # Create nodes dictionary
        nodes_dict = {node['id']: node for node in self.research_data['thought_nodes']}
        
        if root_id not in nodes_dict:
            print(f"Root node {root_id} not found in thought nodes.")
            return
        
        # Convert thought chain to text
        from thought_chain import thoughtchain_to_text
        chain_text = thoughtchain_to_text(nodes_dict, root_id)
        
        if not chain_text:
            print("No text could be extracted from the thought chain.")
            return
        
        # Prepare the prompt
        additional_context = arg.strip() if arg else ""
        
        main_question = self.research_data.get('main_question', 'the research question')
        
        if additional_context:
            prompt = f"""Your job is to continue the reasoning chain with one additional thought. Do not add more than one thought.
            Follow the chain of thought and add your next thought. Do not add any other text. 
            Keep the length of the thought to be of similar length to the prior thoughts.
            
            User: {main_question}

            Useful information: 
            {additional_context}

            Prior Thoughts:
            {chain_text}
        """
            
        else:
            prompt = f"""Your job is to continue the reasoning chain with one additional thought. Do not add more than one thought.
            Follow the chain of thought and add your next thought. Do not add any other text. 
            Keep the length of the thought to be of similar length to the prior thoughts.
            
            User: {main_question}

            Prior Thoughts:
            {chain_text}
        """
        
        print("Sending thought chain to deepseek-V3 for continued reasoning...")
        print(f"Chain length: {len(chain_text)} characters")
        
        try:
            # Import required modules
            from query_atom_smallworld import call_agent
            
            # Call the reasoning_continuation agent
            response = call_agent(
                agent_type="reasoning_continuation",
                user_prompt=prompt
            )
            
            print("\n" + "="*80)
            print("CONTINUED REASONING:")
            print("="*80)
            print(response)
            print("="*80)
            
            # Optionally add the continued reasoning as new nodes
            add_to_chain = input("\nAdd this continued reasoning to the thought chain? (y/n): ").strip().lower()
            if add_to_chain == 'y':
                # Convert the response to new thought nodes and add to chain
                from thought_chain import thoughts_text_to_nodes
                new_nodes = thoughts_text_to_nodes(response)
                
                if new_nodes:
                    # Find the last node in the current chain
                    current_id = root_id
                    while current_id is not None:
                        node = nodes_dict[current_id]
                        children = node.get('children_ids', [])
                        if children:
                            current_id = children[0]  # Follow the chain
                        else:
                            break  # Found the end
                    
                    # Connect first new node to the end of current chain
                    if current_id and new_nodes:
                        first_new_node = new_nodes[0]
                        first_new_node.parent_id = current_id
                        nodes_dict[current_id]['children_ids'] = [first_new_node.id]
                    
                    # Add all new nodes to the research data
                    for node in new_nodes:
                        self.research_data['thought_nodes'].append(node.to_dict())
                    
                    print(f"Added {len(new_nodes)} new reasoning nodes to the thought chain.")
                    self._save_session()
                else:
                    print("Could not parse the continued reasoning into thought nodes.")
            
        except Exception as e:
            print(f"Error during reasoning continuation: {str(e)}")

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
