import os
import json
import sqlite3
import pickle
import time
import datetime
import hashlib
from typing import Dict, List, Any, Optional

class SessionManager:
    """Manages research sessions, databases, and data persistence."""
    
    def __init__(self, base_dir="./research_sessions"):
        """Initialize the session manager with base directory for sessions."""
        self.base_dir = base_dir
        self.sessions_dir = base_dir
        
        # Import necessary functionality from query_atom_smallworld
        try:
            from query_atom_smallworld import DEFAULT_CONFIG, SCIENTIFIC_RELATIONS, FIRECRAWL_CONFIG
            self.default_config = DEFAULT_CONFIG
            self.scientific_relations = SCIENTIFIC_RELATIONS
            self.firecrawl_config = FIRECRAWL_CONFIG
        except ImportError:
            print("Warning: Could not import from query_atom_smallworld. Some functionality may be limited.")
            self.default_config = {}
            self.scientific_relations = {}
            self.firecrawl_config = {}
        
        # Create only the base directory if it doesn't exist
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        # Session and research data
        self.current_session = None
        self.research_data = None
        self.facts = []
        self.deleted_facts = []
        self.databases = None
        self.vector_store = None  # Initialize vector_store attribute
        
        # Load available sessions
        self.sessions = self._get_available_sessions()
    
    def _get_available_sessions(self) -> Dict[str, Dict]:
        """Get a dictionary of available research sessions with metadata."""
        sessions = {}
        
        if not os.path.exists(self.sessions_dir):
            return sessions
            
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith('.json'):
                # Skip deleted facts files which contain a list instead of a session dictionary
                if filename.endswith('_deleted.json'):
                    continue
                    
                session_id = filename.replace('.json', '')
                try:
                    with open(os.path.join(self.sessions_dir, filename), 'r') as f:
                        data = json.load(f)
                        
                    # Check if it's a deleted facts file (list)
                    if isinstance(data, list):
                        # Skip this file or handle it specially
                        continue
                        
                    # Extract basic metadata
                    sessions[session_id] = {
                        'question': data.get('main_question', 'Unknown'),
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'fact_count': len(data.get('facts', [])),
                    }
                except Exception as e:
                    # Silent failure but we should log it
                    print(f"Error loading session {session_id}: {str(e)}")
                    
        return sessions
    
    def load_session(self, session_id: str) -> bool:
        """Load a research session by ID."""
        if not session_id:
            print("Please specify a session ID.")
            return False
            
        # Check if the ID is a deleted facts file
        if session_id.endswith("_deleted"):
            print(f"'{session_id}' is a deleted facts archive, not a valid session.")
            print("Please use a regular session ID instead.")
            return False
            
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            print(f"Session '{session_id}' not found.")
            return False
            
        try:
            # Load session data
            with open(session_file, 'r', encoding='utf-8') as f:
                self.research_data = json.load(f)
                
            # Set up session and facts
            self.current_session = session_id
            self.facts = self.research_data.get('facts', [])
            
            # Get deleted facts from the research_data
            self.deleted_facts = self.research_data.get('deleted_facts', [])
            
            # Initialize database connections
            self._initialize_databases(session_id)
            
            print(f"Loaded session: {session_id}")
            print(f"Question: {self.research_data.get('main_question', 'Unknown')}")
            print(f"Facts: {len(self.facts)}")
            if self.deleted_facts:
                print(f"Deleted facts: {len(self.deleted_facts)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            return False
    
    def save_session(self) -> bool:
        """Save the current research session to disk."""
        if not self.current_session or not self.research_data:
            print("No active session to save.")
            return False
        
        try:
            # Ensure the session directory exists
            os.makedirs(self.sessions_dir, exist_ok=True)
            
            # Make sure facts are up to date in the research data
            self.research_data['facts'] = self.facts
            
            # Save main session file
            session_file = os.path.join(self.sessions_dir, f"{self.current_session}.json")
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.research_data, f, indent=2, default=str)
            
            # Separately save deleted facts (as in original code)
            if self.deleted_facts:
                deleted_file = os.path.join(self.sessions_dir, f"{self.current_session}_deleted.json")
                try:
                    with open(deleted_file, 'w', encoding='utf-8') as f:
                        json.dump(self.deleted_facts, f, indent=2, default=str)
                except Exception as e:
                    print(f"Error saving deleted facts: {str(e)}")
            
            print(f"Session saved successfully as '{self.current_session}'")
            return True
            
        except Exception as e:
            print(f"Error saving session: {str(e)}")
            return False
    
    def create_session(self, question, research_data):
        """Create a new research session."""
        try:
            # Generate a session ID
            session_id = self._generate_session_id(question)
            
            # Set up session data
            self.current_session = session_id
            self.research_data = research_data
            
            # Make sure research_data has 'main_question'
            if isinstance(research_data, dict) and 'main_question' not in research_data:
                self.research_data['main_question'] = question
            
            # Extract facts - use facts directly from research_data if available
            if isinstance(research_data, dict) and 'facts' in research_data and isinstance(research_data['facts'], list):
                self.facts = research_data['facts']
            else:
                # If no facts available, initialize empty list
                self.facts = []
            
            # Set up other session properties
            self.deleted_facts = []
            
            # Initialize databases
            self._initialize_databases(session_id)
            
            # Create vector store from the search results
            self._build_vector_store_from_research_data()
            
            # Save session data
            self.save_session()
            
            # Update sessions list
            self.sessions = self._get_available_sessions()
            
            return session_id
            
        except Exception as e:
            print(f"Error creating session: {str(e)}")
            return None
    
    def _generate_session_id(self, question):
        """
        Generate a unique session ID based on the research question.
        
        Args:
            question: The main research question
            
        Returns:
            A unique session identifier string
        """
        # Create a timestamp component for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a hash of the question for a consistent ID component
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        
        # Combine timestamp and question hash for the session ID
        session_id = f"{timestamp}_{question_hash}"
        
        return session_id
    
    def _build_vector_store_from_research_data(self):
        """Build a vector store from chunks in the research data after searches complete."""
        try:
            # Extract all chunks from search results
            all_chunks = []
            
            # Function to recursively extract chunks from research data and followups
            def extract_chunks(data):
                chunks = []
                # Extract from main search metadata if available
                if 'search_metadata' in data and 'sources' in data['search_metadata']:
                    for source in data['search_metadata']['sources']:
                        if 'chunks' in source:
                            for chunk in source['chunks']:
                                chunks.append({
                                    'text': chunk['text'],
                                    'source_id': source['id'],
                                    'chunk_id': chunk.get('id', ''),
                                    'url': source.get('url', ''),
                                    'title': source.get('title', '')
                                })
                
                # Process followups recursively
                if 'followups' in data and data['followups']:
                    for followup in data['followups']:
                        chunks.extend(extract_chunks(followup))
                return chunks
                
            # Extract chunks from research data
            all_chunks = extract_chunks(self.research_data)
            
            if not all_chunks:
                print("No chunks found in research data")
                return
            
            # Get embedder and create vector store
            from embeddings import get_embedder
            embedder = get_embedder()
            self.vector_store = embedder.create_vector_store(all_chunks)
            
        except Exception as e:
            print(f"Error building vector store: {str(e)}")
            self.vector_store = None
    
    def _initialize_databases(self, session_id: str) -> None:
        """Initialize database connections for the session."""
        db_path = os.path.join(self.sessions_dir, f"{session_id}.db")
        
        # Create a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        
        # Create tables if they don't exist
        try:
            cursor = conn.cursor()
            
            # Create facts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                relation TEXT,
                object TEXT,
                source_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create sources table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                title TEXT,
                url TEXT,
                content TEXT,
                type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create embeddings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                embedding BLOB,
                type TEXT,
                reference_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create chat history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
            ''')
            
            conn.commit()
            
            # Store the database connections
            self.databases = {
                'relational': conn
            }
            
            # Populate the database with facts if it's empty
            cursor.execute("SELECT COUNT(*) FROM facts")
            fact_count = cursor.fetchone()[0]
            
            if fact_count == 0 and self.facts:
                print(f"Initializing database with {len(self.facts)} facts...")
                for fact in self.facts:
                    subject = fact.get('subject', '')
                    relation = fact.get('relation', '')
                    obj = fact.get('object', '')
                    source_id = 'unknown'
            
                    
                    if 'evidence' in fact and fact['evidence']:
                        source_id = fact['evidence'][0].get('source_id', 'unknown')
                    
                    cursor.execute(
                        'INSERT INTO facts (subject, relation, object, source_id) VALUES (?, ?, ?, ?)',
                        (subject, relation, obj, source_id)
                    )
                
                conn.commit()
                print("Database initialized.")
            
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            
    def close(self) -> None:
        """Close all connections and resources."""
        if self.databases:
            for db_name, conn in self.databases.items():
                try:
                    conn.close()
                except:
                    pass
                    
        self.databases = None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a research session by ID.
        
        Args:
            session_id: The ID of the session to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not session_id or session_id not in self.sessions:
            print(f"Session '{session_id}' not found.")
            return False
        
        try:
            # Get the session file path
            session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
            
            # Close database connections if this is the current session
            if self.current_session == session_id and self.databases:
                for db_name, conn in self.databases.items():
                    if hasattr(conn, 'close'):
                        try:
                            conn.close()
                        except Exception:
                            pass
            
            # Delete the session JSON file
            if os.path.exists(session_file):
                os.remove(session_file)
                print(f"Deleted session file: {session_file}")
            
            # Delete the entire session directory if it exists
            session_dir = os.path.join(self.sessions_dir, session_id)
            if os.path.exists(session_dir) and os.path.isdir(session_dir):
                import shutil
                shutil.rmtree(session_dir)
                print(f"Deleted session directory: {session_dir}")
            
            # Reset current session if it was the deleted one
            if self.current_session == session_id:
                self.current_session = None
                self.research_data = None
                self.facts = []
                self.deleted_facts = []
                self.databases = None
                self.vector_store = None
                print("Current session reset as it was deleted.")
            
            # Update sessions list
            self.sessions = self._get_available_sessions()
            print(f"Session '{session_id}' deleted successfully.")
            return True
            
        except Exception as e:
            print(f"Error deleting session '{session_id}': {str(e)}")
            return False
