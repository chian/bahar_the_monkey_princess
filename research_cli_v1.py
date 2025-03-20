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

# Import the core functions from your existing script
from query_atom_smallworld import run_scientific_reasoning_workflow, SCIENTIFIC_RELATIONS, FIRECRAWL_CONFIG

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
        self.session_dir = "research_sessions"
        self.current_session = None
        self.databases = None
        self.research_data = None
        self.facts = []
        self.deleted_facts = []  # Store deleted facts here
        
        # Create sessions directory if it doesn't exist
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
            
        # Check for existing sessions
        self.sessions = self._get_available_sessions()
        if self.sessions:
            print(f"Found {len(self.sessions)} existing research sessions.")
            print("Use 'load <session_id>' to continue a previous session.")
    
    def _get_available_sessions(self) -> Dict[str, str]:
        """Get available saved sessions."""
        sessions = {}
        if os.path.exists(self.session_dir):
            for file in os.listdir(self.session_dir):
                if file.endswith('.json'):
                    # Get clean session ID without .json extension and any quotes
                    session_id = file.replace('.json', '').strip('"').strip("'")
                    
                    # Load basic metadata
                    try:
                        with open(os.path.join(self.session_dir, file), 'r') as f:
                            data = json.load(f)
                            question = data.get('main_question', 'Unknown question')
                            # Remove any quotes from the question text
                            question = question.strip('"').strip("'")
                            
                            # Check if we already have a short title
                            if 'short_title' not in data:
                                short_title = self._generate_short_title(question)
                                # Save the short title back to the session file
                                data['short_title'] = short_title
                                with open(os.path.join(self.session_dir, file), 'w') as f:
                                    json.dump(data, f, indent=2)
                            else:
                                short_title = data['short_title']
                            
                            sessions[session_id] = short_title
                    except Exception as e:
                        print(f"Error processing session {session_id}: {str(e)}")
                        sessions[session_id] = 'Error loading session metadata'
        return sessions
    
    def _generate_short_title(self, question: str, max_length: int = 50) -> str:
        """Generate a short, descriptive title for a research question using LLM."""
        try:
            from query_atom_smallworld import DEFAULT_CONFIG
            import openai
            
            # Return a truncated version if we can't make the API call
            if not hasattr(self, '_openai_client'):
                try:
                    self._openai_client = openai.OpenAI(api_key=DEFAULT_CONFIG["api_key"])
                except:
                    return question[:max_length-3] + "..." if len(question) > max_length else question
            
            # Prompt the model to create a concise title
            response = self._openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a smaller, faster model for this simple task
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise titles."},
                    {"role": "user", "content": f"Create a short, descriptive title (maximum {max_length} characters) for this research question: {question}"}
                ],
                max_tokens=60,
                temperature=0.3
            )
            
            short_title = response.choices[0].message.content.strip().strip('"')
            
            # Ensure the title is within the maximum length
            if len(short_title) > max_length:
                short_title = short_title[:max_length-3] + "..."
            
            return short_title
            
        except Exception as e:
            print(f"Error generating short title: {str(e)}")
            # Fallback to truncation
            return question[:max_length-3] + "..." if len(question) > max_length else question
    
    def _initialize_databases(self):
        """Initialize the databases for storing research information."""
        if self.current_session is None:
            print("No active session. Start a new research or load existing session.")
            return False
            
        session_path = os.path.join(self.session_dir, self.current_session)
        os.makedirs(session_path, exist_ok=True)
        
        self.databases = {
            # SQLite for structured data (facts, sources)
            'relational': sqlite3.connect(os.path.join(session_path, 'research.db')),
            
            # Vector database for semantic search
            'vector': {
                'index': None,
                'metadata': []
            },
            
            # Document storage for full text
            'documents': []
        }
        
        # Initialize SQLite tables
        cursor = self.databases['relational'].cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            relation TEXT,
            object TEXT,
            source_id TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            content TEXT,
            query TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
        ''')
        
        self.databases['relational'].commit()
        
        # Initialize vector database
        dim = 1536  # Default dimension for embeddings
        self.databases['vector']['index'] = faiss.IndexFlatL2(dim)
        
        return True
    
    def _save_session(self):
        """Save the current research session to disk."""
        if not self.current_session or not self.research_data:
            print("No active session to save.")
            return
        
        # Create session directory if it doesn't exist
        session_dir = os.path.join(self.session_dir, self.current_session)
        os.makedirs(session_dir, exist_ok=True)
        
        # Ensure all facts have properly structured evidence
        if 'facts' in self.research_data:
            # Make a deep copy to avoid modifying during iteration
            import copy
            facts_copy = copy.deepcopy(self.research_data['facts'])
            
            for fact in facts_copy:
                # Check if evidence exists and is properly formed
                if 'evidence' not in fact or not fact['evidence']:
                    # Create empty evidence array if missing
                    fact['evidence'] = [{}]
                    
                # Ensure each evidence item has a source_id
                for evidence in fact['evidence']:
                    if 'source_id' not in evidence or not evidence['source_id']:
                        evidence['source_id'] = 'unknown'
                        
                    # Convert any non-serializable objects to strings
                    for key, value in list(evidence.items()):
                        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            evidence[key] = str(value)
            
            # Update the research data with the validated facts
            self.research_data['facts'] = facts_copy
        
        # Save complete research data with pretty-printing for readability
        session_file = os.path.join(self.session_dir, f"{self.current_session}.json")
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.research_data, f, indent=2, default=str)
            
            # Verify the saved data to ensure integrity
            with open(session_file, 'r', encoding='utf-8') as f:
                test_load = json.load(f)
                
            # Check if facts were preserved correctly
            if 'facts' in self.research_data and 'facts' in test_load:
                if len(test_load['facts']) != len(self.research_data['facts']):
                    print("Warning: Some facts may not have been saved correctly.")
            
            print(f"Session saved successfully as '{self.current_session}'")
        except Exception as e:
            print(f"Error saving session: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Save deleted facts if we have any
        if self.deleted_facts:
            deleted_file = os.path.join(session_dir, 'deleted_facts.json')
            try:
                with open(deleted_file, 'w', encoding='utf-8') as f:
                    json.dump(self.deleted_facts, f, indent=2, default=str)
            except Exception as e:
                print(f"Error saving deleted facts: {str(e)}")
        
        # Save vector database if we have embeddings
        if self.databases and 'vector' in self.databases and self.databases['vector']['index'] is not None:
            vector_path = os.path.join(session_dir, 'vector_index.faiss')
            if self.databases['vector']['index'].ntotal > 0:
                faiss.write_index(self.databases['vector']['index'], vector_path)
            
            # Save vector metadata
            meta_path = os.path.join(session_dir, 'vector_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(self.databases['vector']['metadata'], f, indent=2, default=str)
    
    def _load_facts_from_research_data(self):
        """Extract facts from research data and store in database."""
        if not self.research_data or not self.databases:
            return
            
        # Extract all facts from the research data
        facts = []
        if 'facts' in self.research_data:
            facts.extend(self.research_data['facts'])
        
        # Also get facts from followups if they exist
        def extract_followup_facts(results):
            extracted = []
            if 'facts' in results:
                extracted.extend(results['facts'])
            if 'followups' in results:
                for followup in results['followups']:
                    extracted.extend(extract_followup_facts(followup))
            return extracted
            
        if 'results' in self.research_data:
            more_facts = extract_followup_facts(self.research_data['results'])
            facts.extend(more_facts)
        
        # Store facts in the database
        cursor = self.databases['relational'].cursor()
        for fact in facts:
            # Clean up fact data
            subject = fact.get('subject', '')
            relation = fact.get('relation', '')
            obj = fact.get('object', '')
            
            # Get source info if available
            source_id = "unknown"
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
            
            # Calculate a simple confidence score (could be more sophisticated)
            confidence = 0.8  # Default confidence
            
            # Insert fact into database
            cursor.execute(
                'INSERT INTO facts (subject, relation, object, source_id) VALUES (?, ?, ?, ?)',
                (subject, relation, obj, source_id)
            )
        
        # Save sources too
        if 'bibliography' in self.research_data:
            for source in self.research_data['bibliography']:
                source_id = str(source.get('id', ''))
                title = source.get('title', 'Unknown source')
                url = source.get('url', '')
                
                # Combine all chunks into content
                content = ""
                if 'chunks' in source:
                    for chunk in source['chunks']:
                        content += chunk.get('text', '') + "\n\n"
                
                # Get query if available
                query = ""
                if 'chunks' in source and source['chunks']:
                    query = source['chunks'][0].get('query', '')
                
                cursor.execute(
                    'INSERT OR REPLACE INTO sources (id, title, url, content, query) VALUES (?, ?, ?, ?, ?)',
                    (source_id, title, url, content, query)
                )
        
        self.databases['relational'].commit()
        
        # Cache facts for quick access
        self.facts = facts
        print(f"Loaded {len(facts)} facts into the database")
    
    def _generate_embeddings(self, texts):
        """Generate embeddings for text using OpenAI's API."""
        import openai
        
        # This assumes you're using OpenAI - adjust for other embedding sources
        client = openai.OpenAI(api_key=DEFAULT_CONFIG["api_key"])
        
        # Batch process to avoid rate limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(input=batch, model="text-embedding-3-small")
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
    
    def do_research(self, arg):
        """
        Start a new research session with a scientific question.
        Usage: research <scientific question>
        """
        if not arg:
            print("Please provide a scientific question to research.")
            return
            
        # Create a new session
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        question_slug = arg[:20].lower().replace(' ', '_')
        self.current_session = f"{question_slug}_{timestamp}"
        
        print(f"\nStarting research on: {arg}")
        print("This may take several minutes...\n")
        
        # Run the scientific reasoning workflow from the imported module
        try:
            conclusion = run_scientific_reasoning_workflow(
                scientific_question=arg,
                breadth=4,
                depth=2
            )
            
            # The workflow should have saved JSON data we can load
            data_filename = f"research_data_{arg[:30].replace(' ', '_').replace('?', '').replace('/', '_')}.json"
            
            if os.path.exists(data_filename):
                with open(data_filename, 'r', encoding='utf-8') as f:
                    self.research_data = json.load(f)
                
                # After getting results back, fix the fact evidence structure:
                if 'facts' in self.research_data:
                    # Build source lookup from bibliography
                    source_lookup = {}
                    if 'bibliography' in self.research_data:
                        for source in self.research_data['bibliography']:
                            source_id = str(source.get('id', ''))
                            source_lookup[source_id] = {
                                'url': source.get('url', ''),
                                'title': source.get('title', 'Unknown source')
                            }
                    
                    # Ensure each fact's evidence includes URL information
                    for fact in self.research_data['facts']:
                        if 'evidence' in fact and fact['evidence']:
                            for evidence in fact['evidence']:
                                source_id = evidence.get('source_id', '')
                                
                                # If we have this source in bibliography but no URL in evidence, add it
                                if source_id in source_lookup and 'url' not in evidence:
                                    evidence['url'] = source_lookup[source_id]['url']
                
                # Initialize databases
                if self._initialize_databases():
                    # Process the research data into our databases
                    self._load_facts_from_research_data()
                    
                    # Also save the session data
                    self._save_session()
                    
                    print("\nResearch complete! You can now:")
                    print("- Use 'facts' to see extracted facts")
                    print("- Use 'chat' to begin asking questions about the research")
                    print("- Use 'save' to explicitly save this session")
                    print("- Use 'search <query>' to find relevant information")
            else:
                print(f"Warning: Research completed but couldn't find data file {data_filename}")
                
        except Exception as e:
            print(f"Error during research: {str(e)}")
    
    def do_load(self, arg):
        """
        Load an existing research session.
        Usage: load <session_id>
        """
        if not arg:
            # Show available sessions
            print("Available sessions:")
            for i, (session_id, title) in enumerate(self.sessions.items(), 1):
                print(f"  {i}. {session_id} - {title}")
            return
        
        # Clean up the input argument - remove quotes and trim whitespace
        arg = arg.strip().strip('"').strip("'")
        
        # Check if the user entered a number
        try:
            session_num = int(arg)
            if 1 <= session_num <= len(self.sessions):
                # Convert number to session ID
                arg = list(self.sessions.keys())[session_num - 1]
            else:
                print(f"Invalid session number. Please use a number between 1 and {len(self.sessions)}.")
                return
        except ValueError:
            # Not a number, continue with string handling
            pass
        
        # Check if the user entered the full display string with the title
        if " - " in arg:
            # Extract just the session ID part
            arg = arg.split(" - ")[0].strip()
        
        if arg not in self.sessions:
            print(f"Session '{arg}' not found. Use 'load' without arguments to see available sessions.")
            return
        
        # Load the session data
        try:
            session_file = os.path.join(self.session_dir, f"{arg}.json")
            with open(session_file, 'r', encoding='utf-8') as f:
                self.research_data = json.load(f)
                
            self.current_session = arg
            print(f"Loaded research session: {self.research_data.get('main_question', 'Unknown question')}")
            
            # Initialize databases
            if self._initialize_databases():
                # Check if we have existing DB files
                db_path = os.path.join(self.session_dir, self.current_session)
                
                # Try to load vector database if it exists
                vector_path = os.path.join(db_path, 'vector_index.faiss')
                if os.path.exists(vector_path):
                    self.databases['vector']['index'] = faiss.read_index(vector_path)
                    
                    # Also load metadata
                    meta_path = os.path.join(db_path, 'vector_metadata.json')
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            self.databases['vector']['metadata'] = json.load(f)
                
                # Load deleted facts if they exist
                deleted_file = os.path.join(db_path, 'deleted_facts.json')
                if os.path.exists(deleted_file):
                    with open(deleted_file, 'r') as f:
                        self.deleted_facts = json.load(f)
                    print(f"Loaded {len(self.deleted_facts)} previously deleted facts. Use 'trash' to view them.")
                else:
                    self.deleted_facts = []
                
                # Load facts from research data
                self._load_facts_from_research_data()
                
                print("Session loaded successfully. Use 'chat' to start asking questions.")
        except Exception as e:
            print(f"Error loading session: {str(e)}")
    
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
        Display a numbered list of all facts for easy reference.
        Usage: list_facts [<relation_filter>]
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
        
        # Print facts with numbers
        for i, fact in enumerate(filtered_facts, 1):
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
            
            print(f"  {i}. {status_text}{fact.get('subject', '')} {relation} {fact.get('object', '')}")
    
    def do_delete_fact(self, arg):
        """
        Delete a fact by its number in the list.
        Usage: delete_fact <fact_number>
        """
        if not self.facts:
            print("No facts available. Start a research session first.")
            return
        
        if not arg:
            print("Please specify a fact number to delete.")
            print("Use 'list_facts' to see facts with their numbers.")
            return
        
        try:
            fact_num = int(arg)
            if fact_num < 1 or fact_num > len(self.facts):
                print(f"Invalid fact number. Please use a number between 1 and {len(self.facts)}.")
                return
            
            # Get the fact to delete
            fact_to_delete = self.facts[fact_num - 1]
            
            # Display the fact for confirmation
            relation = fact_to_delete.get('relation', 'unknown')
            print(f"Deleting fact: {fact_to_delete.get('subject', '')} {relation} {fact_to_delete.get('object', '')}")
            
            confirmation = input("Are you sure you want to delete this fact? (y/n): ")
            if confirmation.lower() != 'y':
                print("Deletion cancelled.")
                return
            
            # Delete from the database if we have one
            if self.databases:
                cursor = self.databases['relational'].cursor()
                subject = fact_to_delete.get('subject', '')
                relation = fact_to_delete.get('relation', '')
                obj = fact_to_delete.get('object', '')
                
                cursor.execute(
                    'DELETE FROM facts WHERE subject = ? AND relation = ? AND object = ?',
                    (subject, relation, obj)
                )
                self.databases['relational'].commit()
            
            # Store the deleted fact before removing it
            self.deleted_facts.append(fact_to_delete)
            
            # Remove from the facts list
            del self.facts[fact_num - 1]
            
            # Update research data if it exists
            if self.research_data and 'facts' in self.research_data:
                for i, fact in enumerate(self.research_data['facts'][:]):
                    if (fact.get('subject') == fact_to_delete.get('subject') and
                        fact.get('relation') == fact_to_delete.get('relation') and
                        fact.get('object') == fact_to_delete.get('object')):
                        del self.research_data['facts'][i]
                        break
            
            # Save the session with updated facts
            self._save_session()
            
            print("Fact deleted successfully. Use 'trash' to see deleted facts or 'restore' to recover them.")
            
        except ValueError:
            print("Please enter a valid number.")
    
    def do_add_fact(self, arg):
        """
        Add a new fact to the research.
        Usage: add_fact
        """
        if not self.current_session or not self.research_data:
            print("No active research session. Start or load a session first.")
            return
        
        print("Adding a new fact. Please provide the following information:")
        
        # Show available relation types
        print("\nAvailable relation types:")
        for i, (relation, description) in enumerate(SCIENTIFIC_RELATIONS.items(), 1):
            print(f"  {i}. {relation}: {description}")
        
        # Get fact components from the user
        subject = input("\nSubject: ").strip()
        if not subject:
            print("Subject cannot be empty. Cancelled.")
            return
        
        relation_input = input("Relation (enter number or name): ").strip()
        try:
            # Check if it's a number
            relation_num = int(relation_input)
            if 1 <= relation_num <= len(SCIENTIFIC_RELATIONS):
                relation = list(SCIENTIFIC_RELATIONS.keys())[relation_num - 1]
            else:
                print(f"Invalid relation number. Please use 1-{len(SCIENTIFIC_RELATIONS)}.")
                return
        except ValueError:
            # Not a number, check if it's a valid relation name
            if relation_input in SCIENTIFIC_RELATIONS:
                relation = relation_input
            else:
                print(f"Unknown relation type: {relation_input}")
                return
        
        obj = input("Object: ").strip()
        if not obj:
            print("Object cannot be empty. Cancelled.")
            return
        
        # Create the new fact
        new_fact = {
            'subject': subject,
            'relation': relation,
            'object': obj,
            'confidence': 0.8,  # Default confidence
            'evidence': [{'source_id': 'user_added'}]
        }
        
        # Add to the facts list
        self.facts.append(new_fact)
        
        # Add to the database
        if self.databases:
            cursor = self.databases['relational'].cursor()
            cursor.execute(
                'INSERT INTO facts (subject, relation, object, source_id) VALUES (?, ?, ?, ?)',
                (subject, relation, obj, 'user_added')
            )
            self.databases['relational'].commit()
        
        # Add to research data
        if 'facts' not in self.research_data:
            self.research_data['facts'] = []
        self.research_data['facts'].append(new_fact)
        
        # Save the session
        self._save_session()
        
        print(f"New fact added: {subject} {relation} {obj}")
    
    def do_rethink(self, arg):
        """
        Re-analyze the research question with the current set of facts.
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
        else:
            print(f"Re-analyzing the research question: {question}")
            full_question = question
        
        # Prepare context with all facts
        facts_context = ""
        for i, fact in enumerate(self.facts, 1):
            relation = fact.get('relation', 'unknown')
            facts_context += f"{i}. {fact.get('subject', '')} {relation} {fact.get('object', '')}\n"
        
        # Use OpenAI to generate a new conclusion
        try:
            from query_atom_smallworld import DEFAULT_CONFIG
            import openai
            
            if not hasattr(self, '_openai_client'):
                self._openai_client = openai.OpenAI(api_key=DEFAULT_CONFIG["api_key"])
            
            prompt = f"""Based on the following facts, please re-analyze this research question:
Question: {full_question}

Facts:
{facts_context}

Previous conclusion:
{self.research_data.get('conclusion', 'No previous conclusion.')}

Please provide:
1. A comprehensive analysis
2. A revised conclusion
3. Any new insights or hypotheses
"""
            
            print("\nGenerating new analysis. This may take a moment...\n")
            
            response = self._openai_client.chat.completions.create(
                model="gpt-4.5-preview",  # Using a strong model for analysis
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
            
            # Save the session
            self._save_session()
            
            # Print the results
            print("\n=== New Analysis ===\n")
            print(new_analysis)
            
        except Exception as e:
            print(f"Error generating new analysis: {str(e)}")
    
    def do_relations(self, arg):
        """
        List all available relation types in the scientific data.
        Usage: relations
        """
        print("Available scientific relation types:")
        for relation, description in SCIENTIFIC_RELATIONS.items():
            print(f"- {relation}: {description}")
    
    def do_search(self, arg):
        """
        Search through the research data.
        Usage: search <query>
        """
        if not arg:
            print("Please provide a search query.")
            return
            
        if not self.databases:
            print("No active research session. Use 'research <question>' or 'load <session_id>' first.")
            return
            
        print(f"Searching for: {arg}")
        
        # Search for facts
        cursor = self.databases['relational'].cursor()
        cursor.execute(
            '''SELECT subject, relation, object, source_id FROM facts 
            WHERE subject LIKE ? OR relation LIKE ? OR object LIKE ?''',
            (f'%{arg}%', f'%{arg}%', f'%{arg}%')
        )
        
        results = cursor.fetchall()
        
        if results:
            print(f"\nFound {len(results)} matching facts:")
            for subject, relation, obj, source_id in results:
                print(f"- {subject} {relation} {obj} [Source: {source_id}]")
        else:
            print("No matching facts found.")
            
        # Search in sources
        cursor.execute(
            '''SELECT id, title, url FROM sources 
            WHERE title LIKE ? OR content LIKE ?''',
            (f'%{arg}%', f'%{arg}%')
        )
        
        source_results = cursor.fetchall()
        
        if source_results:
            print(f"\nFound in {len(source_results)} sources:")
            for source_id, title, url in source_results:
                print(f"- [{source_id}] {title}")
                if url:
                    print(f"  URL: {url}")
                    
    def do_save(self, arg):
        """
        Save the current research session.
        Usage: save
        """
        self._save_session()
    
    def do_quit(self, arg):
        """Exit the application."""
        print("Saving session before exit...")
        self._save_session()
        print("Goodbye!")
        return True
        
    # Alias for quit
    do_exit = do_quit
    
    def do_sessions(self, arg):
        """List all available research sessions."""
        # Refresh the list
        self.sessions = self._get_available_sessions()
        
        if not self.sessions:
            print("No saved research sessions found.")
            return
            
        print("Available research sessions:")
        # Add numbers for easy reference
        for i, (session_id, title) in enumerate(self.sessions.items(), 1):
            print(f"  {i}. {session_id} - {title}")
        
        print("\nUse 'load <session_id>' to load a session.")
        print("You can also use the number (e.g., 'load 1').")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass

    def do_delete(self, arg):
        """Delete a research session.
        
        Usage: delete <session_id_or_number>
        """
        if not arg:
            print("Please specify a session ID or number to delete.")
            print("Usage: delete <session_id_or_number>")
            return
        
        # Refresh the list
        self.sessions = self._get_available_sessions()
        
        if not self.sessions:
            print("No saved research sessions found.")
            return
        
        session_id = arg.strip().strip('"').strip("'")
        
        # Check if the user entered a number instead of a session ID
        try:
            session_num = int(session_id)
            if 1 <= session_num <= len(self.sessions):
                # Convert the number to the corresponding session ID
                session_id = list(self.sessions.keys())[session_num - 1]
            else:
                print(f"Invalid session number. Please use a number between 1 and {len(self.sessions)}.")
                return
        except ValueError:
            # Not a number, assume it's a session ID
            if session_id not in self.sessions:
                print(f"Session '{session_id}' not found.")
                return
        
        # Get the session title for confirmation
        session_title = self.sessions[session_id]
        
        # Ask for confirmation
        confirmation = input(f"Are you sure you want to delete '{session_title}' [{session_id}]? (y/n): ")
        if confirmation.lower() != 'y':
            print("Deletion cancelled.")
            return
        
        # Delete the session file
        session_file = os.path.join(self.session_dir, f"{session_id}.json")
        try:
            if os.path.exists(session_file):
                os.remove(session_file)
                print(f"Session '{session_title}' deleted successfully.")
                
                # If the current session is the one we deleted, reset it
                if self.current_session == session_id:
                    self.current_session = None
                    self.research_data = None
                    self.facts = []
                    print("Current session was deleted. Starting a new session.")
                    
                # Refresh the sessions list
                self.sessions = self._get_available_sessions()
            else:
                print(f"Session file not found: {session_file}")
        except Exception as e:
            print(f"Error deleting session: {str(e)}")

    def do_trash(self, arg):
        """
        List all deleted facts that can be restored.
        Usage: trash
        """
        if not self.deleted_facts:
            print("Trash is empty. No deleted facts to restore.")
            return
        
        print(f"Deleted facts (most recent first):")
        for i, fact in enumerate(self.deleted_facts, 1):
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
            
            print(f"  {i}. {status_text}{fact.get('subject', '')} {relation} {fact.get('object', '')}")
        
        print("\nUse 'restore <number>' to restore a fact.")
    
    def do_restore(self, arg):
        """
        Restore a previously deleted fact.
        Usage: restore <fact_number>
        """
        if not self.deleted_facts:
            print("Trash is empty. No deleted facts to restore.")
            return
        
        if not arg:
            print("Please specify a fact number to restore.")
            print("Use 'trash' to see deleted facts with their numbers.")
            return
        
        try:
            fact_num = int(arg)
            if fact_num < 1 or fact_num > len(self.deleted_facts):
                print(f"Invalid fact number. Please use a number between 1 and {len(self.deleted_facts)}.")
                return
            
            # Get the fact to restore
            fact_to_restore = self.deleted_facts[fact_num - 1]
            
            # Display the fact
            relation = fact_to_restore.get('relation', 'unknown')
            print(f"Restoring fact: {fact_to_restore.get('subject', '')} {relation} {fact_to_restore.get('object', '')}")
            
            # Add back to the database
            if self.databases:
                cursor = self.databases['relational'].cursor()
                subject = fact_to_restore.get('subject', '')
                relation = fact_to_restore.get('relation', '')
                obj = fact_to_restore.get('object', '')
                source_id = fact_to_restore.get('evidence', [{}])[0].get('source_id', 'restored')
                confidence = fact_to_restore.get('confidence', 0.8)
                
                cursor.execute(
                    'INSERT INTO facts (subject, relation, object, source_id) VALUES (?, ?, ?, ?)',
                    (subject, relation, obj, source_id)
                )
                self.databases['relational'].commit()
            
            # Add back to the facts list
            self.facts.append(fact_to_restore)
            
            # Update research data
            if self.research_data:
                if 'facts' not in self.research_data:
                    self.research_data['facts'] = []
                self.research_data['facts'].append(fact_to_restore)
            
            # Remove from deleted facts
            del self.deleted_facts[fact_num - 1]
            
            # Save the session
            self._save_session()
            
            print("Fact restored successfully.")
            
        except ValueError:
            print("Please enter a valid number.")

    def do_empty_trash(self, arg):
        """
        Permanently delete all facts in the trash.
        Usage: empty_trash
        """
        if not self.deleted_facts:
            print("Trash is already empty.")
            return
        
        count = len(self.deleted_facts)
        confirmation = input(f"Are you sure you want to permanently delete {count} facts? (y/n): ")
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            return
        
        self.deleted_facts = []
        print(f"Trash emptied. {count} facts permanently deleted.")

    def do_facts_by_source(self, arg):
        """
        Display facts grouped by their source.
        Usage: facts_by_source [<source_id>]
        """
        if not self.facts:
            print("No facts available. Start a research session first.")
            return
        
        # Get all unique source IDs
        sources = set()
        for fact in self.facts:
            source_id = "unknown"
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
            sources.add(source_id)
        
        # If a specific source is requested, filter facts
        if arg:
            if arg not in sources:
                print(f"Source '{arg}' not found. Available sources:")
                for i, source in enumerate(sorted(sources), 1):
                    print(f"  {i}. {source}")
                return
            sources = [arg]
            print(f"Facts from source '{arg}':")
        else:
            print("Facts by source:")
        
        # Display facts grouped by source
        fact_number = 1
        for source_id in sorted(sources):
            source_facts = []
            for fact in self.facts:
                fact_source = "unknown"
                if 'evidence' in fact and fact['evidence']:
                    fact_source = fact['evidence'][0].get('source_id', 'unknown')
                if fact_source == source_id:
                    source_facts.append(fact)
            
            if source_facts:
                print(f"\n== Source: {source_id} ({len(source_facts)}) ==")
                for fact in source_facts:
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

    def do_delete_by_source(self, arg):
        """
        Delete all facts from a specific source.
        Usage: delete_by_source <source_id>
        """
        if not self.facts:
            print("No facts available. Start a research session first.")
            return
        
        if not arg:
            print("Please specify a source ID to delete facts from.")
            print("Use 'facts_by_source' to see available sources.")
            return
        
        # Find facts with the specified source
        facts_to_delete = []
        for fact in self.facts:
            source_id = "unknown"
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
            if source_id == arg:
                facts_to_delete.append(fact)
        
        if not facts_to_delete:
            print(f"No facts found from source '{arg}'.")
            return
        
        # Display facts for confirmation
        print(f"Found {len(facts_to_delete)} facts from source '{arg}':")
        for i, fact in enumerate(facts_to_delete, 1):
            relation = fact.get('relation', 'unknown')
            print(f"  {i}. {fact.get('subject', '')} {relation} {fact.get('object', '')}")
        
        confirmation = input(f"\nAre you sure you want to delete all {len(facts_to_delete)} facts from source '{arg}'? (y/n): ")
        if confirmation.lower() != 'y':
            print("Deletion cancelled.")
            return
        
        # Delete the facts
        deleted_count = 0
        for fact in facts_to_delete[:]:  # Use a copy of the list since we'll modify the original
            # Add to deleted facts for potential restoration
            self.deleted_facts.append(fact)
            
            # Remove from facts list
            self.facts.remove(fact)
            
            # Delete from database
            if self.databases:
                cursor = self.databases['relational'].cursor()
                subject = fact.get('subject', '')
                relation = fact.get('relation', '')
                obj = fact.get('object', '')
                
                cursor.execute(
                    'DELETE FROM facts WHERE subject = ? AND relation = ? AND object = ?',
                    (subject, relation, obj)
                )
                self.databases['relational'].commit()
            
            # Update research data
            if self.research_data and 'facts' in self.research_data:
                for i, data_fact in enumerate(self.research_data['facts'][:]):
                    if (data_fact.get('subject') == fact.get('subject') and
                        data_fact.get('relation') == fact.get('relation') and
                        data_fact.get('object') == fact.get('object')):
                        del self.research_data['facts'][i]
                        break
            
            deleted_count += 1
        
        # Save the session
        self._save_session()
        
        print(f"Deleted {deleted_count} facts from source '{arg}'.")
        print("Use 'trash' to view deleted facts or 'restore' to recover them.")

    def do_list_sources(self, arg):
        """
        List all sources used in the facts.
        Usage: list_sources
        """
        if not self.facts:
            print("No facts available. Start a research session first.")
            return
        
        # Get all unique source IDs and count facts per source, plus verification stats
        source_stats = {}
        source_urls = {}  # Track URLs for each source
        
        for fact in self.facts:
            source_id = "unknown"
            source_url = None
            
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
                source_url = fact['evidence'][0].get('url', None)
                
                # Store URL for this source if we have one
                if source_url and source_id not in source_urls:
                    source_urls[source_id] = source_url
            
            if source_id not in source_stats:
                source_stats[source_id] = {
                    'count': 0,
                    'confirmed': 0,
                    'unverified': 0,
                    'false': 0,
                    'controversial': 0
                }
            
            source_stats[source_id]['count'] += 1
            
            # Count verification status
            if 'verification' in fact:
                status = fact['verification'].get('status', 'unknown')
                if status in ['confirmed', 'unverified', 'false', 'controversial']:
                    source_stats[source_id][status] += 1
        
        print("Available sources:")
        for i, (source_id, stats) in enumerate(sorted(source_stats.items()), 1):
            verification_info = ""
            if stats['confirmed'] > 0 or stats['unverified'] > 0 or stats['false'] > 0 or stats['controversial'] > 0:
                verification_info = f" - Verification: "
                if stats['confirmed'] > 0:
                    verification_info += f"[CONFIRMED: {stats['confirmed']}] "
                if stats['unverified'] > 0:
                    verification_info += f"[UNVERIFIED: {stats['unverified']}] "
                if stats['false'] > 0:
                    verification_info += f"[FALSE: {stats['false']}] "
                if stats['controversial'] > 0:
                    verification_info += f"[WARNING: {stats['controversial']}] "
            
            print(f"  {i}. {source_id} ({stats['count']} facts){verification_info}")
            
            # Display URL if available
            if source_id in source_urls and source_urls[source_id]:
                print(f"     URL: {source_urls[source_id]}")
            
            # Try to get source title if available
            if self.databases and source_id != "unknown" and source_id != "user_added":
                try:
                    cursor = self.databases['relational'].cursor()
                    cursor.execute('SELECT title FROM sources WHERE id = ?', (source_id,))
                    result = cursor.fetchone()
                    if result and result[0]:
                        print(f"     Title: {result[0]}")
                except:
                    pass
        
        print("\nUse 'facts_by_source <source_id>' to see facts from a specific source")
        print("Use 'delete_by_source <source_id>' to delete all facts from a source")
        print("Use 'verify_facts <source_id>' to verify facts from a source")

    def do_fix_sources(self, arg):
        """
        Fix missing source IDs in facts.
        This regenerates proper source identification for facts.
        Usage: fix_sources
        """
        if not self.facts:
            print("No facts to fix.")
            return
        
        fixed_count = 0
        for i, fact in enumerate(self.facts):
            if 'evidence' not in fact or not fact['evidence']:
                fact['evidence'] = [{'source_id': 'regenerated_source'}]
                fixed_count += 1
            else:
                for evidence in fact['evidence']:
                    if 'source_id' not in evidence or not evidence['source_id']:
                        # Try to generate a source ID from other evidence fields
                        if 'chunk_id' in evidence:
                            evidence['source_id'] = f"source_{evidence['chunk_id']}"
                        elif 'query' in evidence:
                            # Hash the query to create a consistent ID
                            evidence['source_id'] = f"query_{hashlib.md5(evidence['query'].encode()).hexdigest()[:8]}"
                        else:
                            # Use a numbered source ID based on position
                            evidence['source_id'] = f"fact_{i+1}_source"
                        
                        fixed_count += 1
        
        if fixed_count > 0:
            print(f"Fixed {fixed_count} source IDs in {len(self.facts)} facts.")
            # Save the session with fixed sources
            self._save_session()
        else:
            print("No source IDs needed fixing.")

    def do_export_json(self, arg):
        """
        Export the current session data to a JSON file using the original format.
        This will ensure compatibility with the query_atom_smallworld.py format.
        Usage: export_json [filename]
        """
        if not self.current_session or not self.research_data:
            print("No active session to export.")
            return
        
        # Use provided filename or generate one
        filename = arg if arg else f"export_{self.current_session}.json"
        
        # Ensure the research_data structure matches query_atom_smallworld.py expectations
        export_data = {
            "main_question": self.research_data.get("main_question", "Unknown question"),
            "conclusion": self.research_data.get("conclusion", ""),
            "timestamp": self.research_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "results": self.research_data.get("results", {}),
            "facts": self.facts,  # Use the current facts list
            "bibliography": self.research_data.get("bibliography", []),
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"Session data exported to {filename}")
        except Exception as e:
            print(f"Error exporting data: {str(e)}")

    def do_verify_facts(self, arg):
        """
        Verify facts from a specific source using Firecrawl search.
        Facts will be classified as 'confirmed', 'unverified', 'false', or 'controversial'.
        
        Usage: verify_facts <source_id>
        """
        if not self.facts:
            print("No facts available. Start a research session first.")
            return
        
        if not arg:
            print("Please specify a source ID to verify facts from.")
            print("Use 'list_sources' to see available sources.")
            return
        
        # Find facts with the specified source and track which ones need verification
        unverified_facts = []
        already_verified = 0
        
        for fact in self.facts:
            source_id = "unknown"
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
                
            if source_id == arg:
                if 'verification' in fact and fact['verification'].get('status') in ['confirmed', 'unverified', 'false', 'controversial']:
                    already_verified += 1
                else:
                    unverified_facts.append(fact)
        
        total_facts = already_verified + len(unverified_facts)
        
        if not unverified_facts:
            if already_verified > 0:
                print(f"All facts from source '{arg}' have already been verified ({already_verified} facts).")
                print("Use 'facts_by_source' to see the verified facts.")
            else:
                print(f"No facts found from source '{arg}'.")
            return
        
        print(f"Found {len(unverified_facts)} facts from source '{arg}' to verify.")
        print("This process may take some time as it involves web searches and analysis.")
        
        confirmation = input("Do you want to proceed with verification? (y/n): ")
        if confirmation.lower() != 'y':
            print("Verification cancelled.")
            return
        
        # Import necessary functions from query_atom_smallworld
        try:
            from query_atom_smallworld import DEFAULT_CONFIG, FIRECRAWL_CONFIG, SCIENTIFIC_RELATIONS
            from firecrawl import FirecrawlApp
            import openai
            
            if not hasattr(self, '_openai_client'):
                self._openai_client = openai.OpenAI(api_key=DEFAULT_CONFIG["api_key"])
                
            firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_CONFIG["api_key"])
            
            # Define function to search with Firecrawl
            def verify_search(query):
                """Search with Firecrawl and return results."""
                print(f"Searching with Firecrawl: {query}")
                try:
                    # Update to match V1 implementation
                    response = firecrawl_app.search(
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
                verification_data = {
                    'fact': fact,
                    'searches': [],
                    'confirming_texts': [],
                    'contradicting_texts': [],
                    'status': 'unverified',
                    'confidence': 0.0
                }
                
                # Use our patched function that uses the correct parameters
                print(f"  Searching with Firecrawl using {len(search_queries)} queries...")
                search_results = verify_search(search_queries[0])
                
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
                        # Firecrawl already provides the content, so no need to scrape
                        content = result.get('content', '')
                        
                        if not content:
                            continue
                        
                        # Analyze if the content confirms or contradicts the fact
                        analysis = self._analyze_content_for_fact(fact, content)
                        
                        if analysis['confirms']:
                            verification_data['confirming_texts'].append({
                                'url': result.get('url', ''),
                                'title': result.get('title', ''),
                                'text': analysis['relevant_text'],
                                'confidence': analysis['confidence']
                            })
                            print(f"  [CONFIRM] Found confirming evidence in: {result.get('title', result.get('url', ''))}")
                        
                        if analysis['contradicts']:
                            verification_data['contradicting_texts'].append({
                                'url': result.get('url', ''),
                                'title': result.get('title', ''),
                                'text': analysis['relevant_text'],
                                'confidence': analysis['confidence']
                            })
                            print(f"  [CONTRADICT] Found contradicting evidence in: {result.get('title', result.get('url', ''))}")
                        
                        # Also extract new facts if possible
                        if analysis['new_facts']:
                            for new_fact in analysis['new_facts']:
                                print(f"  [NEW FACT] Extracted new fact: {new_fact['subject']} {new_fact['relation']} {new_fact['object']}")
                                # Add the new fact with source
                                new_fact['evidence'] = [{
                                    'source_id': f"web_{hashlib.md5(result.get('url', '').encode()).hexdigest()[:8]}",
                                    'url': result.get('url', ''),
                                    'title': result.get('title', ''),
                                    'search_query': result.get('query', '')
                                }]
                                self.facts.append(new_fact)
                                
                                # Also add to research data
                                if self.research_data and 'facts' in self.research_data:
                                    self.research_data['facts'].append(new_fact)
                
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
                    fact['verification'] = {
                        'status': 'controversial',
                        'confirming': len(verification_data['confirming_texts']),
                        'contradicting': len(verification_data['contradicting_texts']),
                        'evidence': verification_data['confirming_texts'] + verification_data['contradicting_texts'],
                        'original_source_id': original_source_id
                    }
                    
                    # Update the fact's primary source to the confirming source
                    if 'evidence' in fact and fact['evidence']:
                        # Store the URL from the best evidence as the new source_id
                        new_source_id = verification_data['confirming_texts'][0].get('url', '')
                        
                        # Update the evidence with the new source info
                        fact['evidence'][0] = {
                            'source_id': new_source_id,
                            'url': new_source_id,
                            'title': verification_data['confirming_texts'][0].get('title', ''),
                            'original_source_id': original_source_id
                        }
                        
                        print(f"  Updated fact source from '{original_source_id}' to '{new_source_id}'")
                        
                        # Also update the fact in research_data if it exists there
                        if self.research_data and 'facts' in self.research_data:
                            # Find the same fact in research_data and update it
                            for research_fact in self.research_data['facts']:
                                if (research_fact.get('subject') == fact.get('subject') and
                                    research_fact.get('relation') == fact.get('relation') and
                                    research_fact.get('object') == fact.get('object')):
                                    # Update the verification and evidence in the research_data copy
                                    research_fact['verification'] = fact['verification']
                                    research_fact['evidence'] = fact['evidence']
                                    break
                
                elif verification_data['contradicting_texts'] and not verification_data['confirming_texts']:
                    verification_data['status'] = 'false'
                    verification_data['confidence'] = 0.9
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
                    
                    # Automatically delete false facts without asking
                    self.facts.remove(fact)
                    
                    # Also remove from research_data if it exists there
                    if self.research_data and 'facts' in self.research_data:
                        # Find and remove the same fact in research_data
                        for i, research_fact in enumerate(self.research_data['facts']):
                            if (research_fact.get('subject') == fact.get('subject') and
                                research_fact.get('relation') == fact.get('relation') and
                                research_fact.get('object') == fact.get('object')):
                                self.research_data['facts'].pop(i)
                                break
                    
                    print("  Automatically deleted false fact.")
                
                elif verification_data['confirming_texts']:
                    verification_data['status'] = 'confirmed'
                    print("  [CONFIRMED] Fact is CONFIRMED")
                    verification_results['confirmed'] += 1
                    
                    # Get the most confident confirming text to use as the new source
                    best_evidence = max(verification_data['confirming_texts'], 
                                       key=lambda x: x.get('confidence', 0))
                    
                    # Update the fact with verification status
                    fact['verification'] = {
                        'status': 'confirmed',
                        'confirming': len(verification_data['confirming_texts']),
                        'contradicting': 0,
                        'evidence': verification_data['confirming_texts'],
                        'original_source_id': fact['evidence'][0].get('source_id') if 'evidence' in fact and fact['evidence'] else 'unknown'
                    }
                    
                    # Update the fact's primary source to the confirming source
                    if 'evidence' in fact and fact['evidence']:
                        # Store the URL from the best evidence as the new source_id
                        new_source_id = best_evidence.get('url', '')
                        
                        # Update the evidence with the new source info
                        fact['evidence'][0] = {
                            'source_id': new_source_id,
                            'url': new_source_id,
                            'title': best_evidence.get('title', ''),
                            'original_source_id': fact['evidence'][0].get('source_id', 'unknown')
                        }
                        
                        print(f"  Updated fact source from '{fact['verification']['original_source_id']}' to '{new_source_id}'")
                        
                        # Also update the fact in research_data if it exists there
                        if self.research_data and 'facts' in self.research_data:
                            # Find the same fact in research_data and update it
                            for research_fact in self.research_data['facts']:
                                if (research_fact.get('subject') == fact.get('subject') and
                                    research_fact.get('relation') == fact.get('relation') and
                                    research_fact.get('object') == fact.get('object')):
                                    # Update the verification and evidence in the research_data copy
                                    research_fact['verification'] = fact['verification']
                                    research_fact['evidence'] = fact['evidence']
                                    break
                
                else:
                    verification_data['status'] = 'unverified'
                    print("  [UNVERIFIED] Fact is UNVERIFIED - could not find confirming or contradicting evidence")
                    verification_results['unverified'] += 1
                
                # Save session after each fact to preserve progress
                self._save_session()
                print("  Progress saved.")
            
            # Print summary
            print("\nVerification Summary:")
            print(f"  [CONFIRMED] Confirmed: {verification_results['confirmed']}")
            print(f"  [UNVERIFIED] Unverified: {verification_results['unverified']}")
            print(f"  [FALSE] False: {verification_results['false']}")
            print(f"  [WARNING] Controversial: {verification_results['controversial']}")
            
            # Save the session with verification results
            self._save_session()

        except ImportError as e:
            print(f"Error importing required functions: {e}")
            return

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
            # Use LLM to analyze if the content confirms, contradicts, or is neutral about the fact
            prompt = f"""Analyze if the following text confirms, contradicts, or is neutral about this scientific fact:

Fact: {subject} {relation} {obj}

Text: {content[:5000]}  # Limit text to first 5000 chars

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
Use the following relation types for new facts: {list(SCIENTIFIC_RELATIONS.keys())}
"""

            response = self._openai_client.chat.completions.create(
                model="gpt-3.5-turbo-1106",  # Fast model with JSON support
                messages=[
                    {"role": "system", "content": "You are a scientific fact-checking assistant. Analyze text and determine if it confirms or contradicts a fact."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Extract and return the analysis
            result['confirms'] = analysis.get('confirms', False)
            result['contradicts'] = analysis.get('contradicts', False)
            result['confidence'] = analysis.get('confidence', 0.0)
            result['relevant_text'] = analysis.get('relevant_text', "")
            result['new_facts'] = analysis.get('new_facts', [])
            
            return result
            
        except Exception as e:
            print(f"  Error analyzing content: {e}")
            return result

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
