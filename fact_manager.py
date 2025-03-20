from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import hashlib

from custom_types import Fact, Source, Evidence, Verification
from research_session import SessionManager

class FactManager:
    """Manages fact operations: listing, adding, deleting, organizing."""
    
    def __init__(self, session_manager: SessionManager):
        """Initialize with a reference to the session manager."""
        self.session_manager = session_manager
        # Initialize fact index
        self._fact_index = set()
        self._rebuild_fact_index()

    def _rebuild_fact_index(self) -> None:
        """Rebuild the fact index for efficient duplicate checking."""
        self._fact_index = {
            self._fact_to_key(fact)
            for fact in self.session_manager.facts
        }

    def _fact_to_key(self, fact: Dict[str, Any]) -> tuple:
        """Convert a fact to a tuple key for indexing."""
        return (
            fact.get('subject', '').lower(),
            fact.get('relation', '').lower(),
            fact.get('object', '').lower()
        )

    def _is_duplicate_fact(self, subject: str, relation: str, obj: str) -> bool:
        """
        Check if a fact with the given subject, relation, and object already exists.
        Uses pre-built index for efficient lookup.
        """
        key = (subject.lower(), relation.lower(), obj.lower())
        return key in self._fact_index

    def list_facts(self, limit: Optional[int] = None, offset: int = 0) -> None:
        """
        List active facts with optional pagination.
        Deleted facts are not included.
        """
        if not self.session_manager.facts:
            print("No facts available. Start a research session first.")
            return
        
        facts_to_display = self.session_manager.facts[offset:]
        if limit:
            facts_to_display = facts_to_display[:limit]
        
        total_active_facts = len(self.session_manager.facts)
        
        print(f"Showing {len(facts_to_display)} facts (from {offset+1} to {offset+len(facts_to_display)} of {total_active_facts}):")
        for i, fact in enumerate(facts_to_display, offset+1):
            relation = fact.get('relation', 'unknown')
            
            # Include verification status if available
            status_text = ""
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
            
            print(f"{i}. {status_text}{fact.get('subject', '')} {relation} {fact.get('object', '')}")
            
            # Show source if available
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
                print(f"   Source: {source_id}")
        
        # Show a hint about deleted facts if there are any
        if self.session_manager.deleted_facts:
            print(f"\n(There are {len(self.session_manager.deleted_facts)} deleted facts. Use 'list_deleted_facts' to view them)")

    def list_deleted_facts(self, limit: Optional[int] = None) -> None:
        """List deleted facts."""
        if not self.session_manager.deleted_facts:
            print("No deleted facts available.")
            return
        
        facts_to_display = self.session_manager.deleted_facts
        if limit:
            facts_to_display = facts_to_display[:limit]
        
        print(f"Showing {len(facts_to_display)} deleted facts (of {len(self.session_manager.deleted_facts)} total):")
        for i, fact in enumerate(facts_to_display, 1):
            relation = fact.get('relation', 'unknown')
            print(f"{i}. {fact.get('subject', '')} {relation} {fact.get('object', '')}")
            
            # Show source if available
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
                print(f"   Source: {source_id}")
        
        print("\nUse 'restore_fact <number>' to restore a deleted fact.")

    def delete_fact(self, index: int) -> None:
        """Delete a fact by its index."""
        if not self.session_manager.facts:
            print("No facts available. Start a research session first.")
            return
        
        try:
            index = int(index) - 1  # Convert to 0-based index
            if index < 0 or index >= len(self.session_manager.facts):
                print(f"Invalid fact number. Please specify a number between 1 and {len(self.session_manager.facts)}.")
                return
            
            fact = self.session_manager.facts[index]
            
            # Remove from fact index
            self._fact_index.remove(self._fact_to_key(fact))
            
            # Add to deleted facts for potential restoration
            self.session_manager.deleted_facts.append(fact)
            
            # Remove from facts list
            del self.session_manager.facts[index]
            
            # Delete from database
            if self.session_manager.databases:
                cursor = self.session_manager.databases['relational'].cursor()
                subject = fact.get('subject', '')
                relation = fact.get('relation', '')
                obj = fact.get('object', '')
                
                cursor.execute(
                    'DELETE FROM facts WHERE subject = ? AND relation = ? AND object = ?',
                    (subject, relation, obj)
                )
                self.session_manager.databases['relational'].commit()
            
            # Update research data
            if self.session_manager.research_data and 'facts' in self.session_manager.research_data:
                for i, data_fact in enumerate(self.session_manager.research_data['facts'][:]):
                    if (data_fact.get('subject') == fact.get('subject') and
                        data_fact.get('relation') == fact.get('relation') and
                        data_fact.get('object') == fact.get('object')):
                        del self.session_manager.research_data['facts'][i]
                        break
            
            # Save the session
            self.session_manager.save_session()
            
            print(f"Deleted fact: {fact.get('subject', '')} {fact.get('relation', '')} {fact.get('object', '')}")
            print("Use 'list_deleted_facts' to view deleted facts or 'restore_fact' to recover them.")
            
        except ValueError:
            print("Please provide a valid fact number.")

    def add_fact(self, fact_str: str) -> None:
        """
        Add a new fact to the current session.
        Format: "<subject> | <relation> | <object>"
        Extended format: "<subject> | <relation> | <object> | <url> | <chunk_id>"
        
        The extended format is primarily used by the system for adding facts with source information.
        User-added facts without source information will be marked as 'unverified' until verified.
        """
        if not self.session_manager.current_session:
            print("No active session. Start a research session first.")
            return
            
        parts = fact_str.split("|")
        if len(parts) not in [3, 5]:
            print("Invalid format. Use either:")
            print("  add_fact <subject> | <relation> | <object>")
            print("  add_fact <subject> | <relation> | <object> | <url> | <chunk_id>")
            print("For example:")
            print("  add_fact Caffeine | increases | alertness")
            print("  add_fact Caffeine | increases | alertness | https://example.com/study | chunk123")
            return
            
        subject = parts[0].strip()
        relation = parts[1].strip()
        obj = parts[2].strip()
        
        # Extract URL and chunk_id if provided
        url = parts[3].strip() if len(parts) > 3 else None
        chunk_id = parts[4].strip() if len(parts) > 4 else None
        
        # Check for duplicates using the helper method
        if self._is_duplicate_fact(subject, relation, obj):
            print(f"Warning: This fact already exists: {subject} {relation} {obj}")
            return
            
        # Create source_id from URL if provided, otherwise use user_added
        source_id = 'user_added'
        if url:
            source_id = f"web_{hashlib.md5(url.encode()).hexdigest()[:8]}"
        
        # Create the new fact with proper typing
        new_fact: Fact = {
            'subject': subject,
            'relation': relation,
            'object': obj,
            'evidence': [
                Evidence(
                    source_id=source_id,
                    text=f"User added fact: {subject} {relation} {obj}",
                    chunk_id=chunk_id,
                    url=url
                )
            ],
            'created_at': datetime.now(),
            'verification': {
                'status': 'unverified',  # Default to unverified
                'confirming': 0,
                'contradicting': 0,
                'evidence': None,
                'original_source_id': None,
                'error': None
            }
        }
        
        # If evidence has URL and chunk_id, mark as confirmed
        if url and chunk_id:
            new_fact['verification']['status'] = 'confirmed'
            new_fact['verification']['confirming'] = 1
            new_fact['verification']['contradicting'] = 0
            new_fact['verification']['evidence'] = [{
                'url': url,
                'title': 'Source provided during fact addition',
                'text': f"Added with source: {url}"
            }]
        
        # Add to facts list
        self.session_manager.facts.append(new_fact)
        
        # Update the fact index
        self._fact_index.add(self._fact_to_key(new_fact))
        
        # Add to database
        if self.session_manager.databases:
            try:
                cursor = self.session_manager.databases['relational'].cursor()
                cursor.execute(
                    'INSERT INTO facts (subject, relation, object, source_id) VALUES (?, ?, ?, ?)',
                    (subject, relation, obj, source_id)
                )
                self.session_manager.databases['relational'].commit()
            except Exception as e:
                print(f"Warning: Could not add fact to database: {e}")
        
        # Update research data
        if self.session_manager.research_data and 'facts' in self.session_manager.research_data:
            self.session_manager.research_data['facts'].append(new_fact)
            
        # Save the session
        self.session_manager.save_session()
        
        verification_status = new_fact['verification']['status']
        print(f"Added new fact: {subject} {relation} {obj} [{verification_status.upper()}]")
        if url:
            print(f"Source: {url}")

    def facts_by_source(self, include_deleted: bool = False) -> None:
        """
        Display facts grouped by their sources.
        By default, only includes active facts (not deleted ones).
        """
        if not self.session_manager.facts:
            print("No facts available. Start a research session first.")
            return
        
        # Group facts by source
        sources: Dict[str, Dict[str, Any]] = {}
        source_urls: Dict[str, Optional[str]] = {}
        
        for fact in self.session_manager.facts:
            if 'evidence' in fact and fact['evidence']:
                source_id = fact['evidence'][0].get('source_id', 'unknown')
                
                if source_id not in sources:
                    sources[source_id] = {
                        'facts': [],
                        'count': 0,
                        'confirmed': 0,
                        'unverified': 0,
                        'false': 0,
                        'controversial': 0
                    }
                
                sources[source_id]['facts'].append(fact)
                sources[source_id]['count'] += 1
                
                # Count verification status
                if 'verification' in fact:
                    status = fact['verification'].get('status', 'unknown')
                    if status == 'confirmed':
                        sources[source_id]['confirmed'] += 1
                    elif status == 'unverified':
                        sources[source_id]['unverified'] += 1
                    elif status == 'false':
                        sources[source_id]['false'] += 1
                    elif status == 'controversial':
                        sources[source_id]['controversial'] += 1
                
                # Extract URL from evidence if available
                if 'url' in fact['evidence'][0] and fact['evidence'][0]['url']:
                    source_urls[source_id] = fact['evidence'][0]['url']
            else:
                # Facts without evidence go to 'unknown' source
                if 'unknown' not in sources:
                    sources['unknown'] = {
                        'facts': [],
                        'count': 0,
                        'confirmed': 0,
                        'unverified': 0,
                        'false': 0,
                        'controversial': 0
                    }
                sources['unknown']['facts'].append(fact)
                sources['unknown']['count'] += 1
        
        # Print sources and fact counts
        print(f"Facts by source ({sum(s['count'] for s in sources.values())} total active facts):")
        for i, (source_id, stats) in enumerate(sorted(sources.items(), key=lambda x: x[1]['count'], reverse=True), 1):
            # Add verification statistics
            verification_info = ""
            if stats['confirmed'] > 0 or stats['unverified'] > 0 or stats['false'] > 0 or stats['controversial'] > 0:
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
            if self.session_manager.databases and source_id != "unknown" and source_id != "user_added":
                try:
                    cursor = self.session_manager.databases['relational'].cursor()
                    cursor.execute('SELECT title FROM sources WHERE id = ?', (source_id,))
                    result = cursor.fetchone()
                    if result and result[0]:
                        print(f"     Title: {result[0]}")
                except:
                    pass
        
        print("\nUse 'facts_by_source <source_id>' to see facts from a specific source")
        print("Use 'delete_by_source <source_id>' to delete all facts from a source")
        print("Use 'verify_facts <source_id>' to verify facts from a source")

    def list_sources(self) -> None:
        """List all sources used in the facts."""
        if not self.session_manager.facts:
            print("No facts available. Start a research session first.")
            return
        
        # Get all unique source IDs and count facts per source, plus verification stats
        source_stats: Dict[str, Dict[str, Any]] = {}
        source_urls: Dict[str, str] = {}
        
        for fact in self.session_manager.facts:
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
            if self.session_manager.databases and source_id != "unknown" and source_id != "user_added":
                try:
                    cursor = self.session_manager.databases['relational'].cursor()
                    cursor.execute('SELECT title FROM sources WHERE id = ?', (source_id,))
                    result = cursor.fetchone()
                    if result and result[0]:
                        print(f"     Title: {result[0]}")
                except:
                    pass
        
        print("\nUse 'facts_by_source <source_id>' to see facts from a specific source")
        print("Use 'delete_by_source <source_id>' to delete all facts from a source")
        print("Use 'verify_facts <source_id>' to verify facts from a source")

    def fix_sources(self) -> None:
        """
        Fix missing source IDs in facts.
        This regenerates proper source identification for facts.
        """
        if not self.session_manager.facts:
            print("No facts to fix.")
            return
        
        fixed_count = 0
        for i, fact in enumerate(self.session_manager.facts):
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
            print(f"Fixed {fixed_count} source IDs in {len(self.session_manager.facts)} facts.")
            # Save the session with fixed sources
            self.session_manager.save_session()
        else:
            print("No source IDs needed fixing.")

    def delete_by_source(self, source_id: str) -> None:
        """
        Delete all facts from a specific source.
        Usage: delete_by_source <source_id>
        """
        if not self.session_manager.facts:
            print("No facts available. Start a research session first.")
            return
        
        if not source_id:
            print("Please specify a source ID to delete facts from.")
            print("Use 'facts_by_source' to see available sources.")
            return
        
        # Find facts with the specified source
        facts_to_delete = []
        for fact in self.session_manager.facts:
            fact_source_id = "unknown"
            if 'evidence' in fact and fact['evidence']:
                fact_source_id = fact['evidence'][0].get('source_id', 'unknown')
            if fact_source_id == source_id:
                facts_to_delete.append(fact)
        
        if not facts_to_delete:
            print(f"No facts found from source '{source_id}'.")
            return
        
        # Display facts for confirmation
        print(f"Found {len(facts_to_delete)} facts from source '{source_id}':")
        for i, fact in enumerate(facts_to_delete, 1):
            relation = fact.get('relation', 'unknown')
            print(f"  {i}. {fact.get('subject', '')} {relation} {fact.get('object', '')}")
        
        confirmation = input(f"\nAre you sure you want to delete all {len(facts_to_delete)} facts from source '{source_id}'? (y/n): ")
        if confirmation.lower() != 'y':
            print("Deletion cancelled.")
            return
        
        # Delete the facts
        deleted_count = 0
        for fact in facts_to_delete[:]:  # Use a copy of the list since we'll modify the original
            # Add to deleted facts for potential restoration
            self.session_manager.deleted_facts.append(fact)
            
            # Remove from facts list
            self.session_manager.facts.remove(fact)
            
            # Delete from database
            if self.session_manager.databases:
                cursor = self.session_manager.databases['relational'].cursor()
                subject = fact.get('subject', '')
                relation = fact.get('relation', '')
                obj = fact.get('object', '')
                
                cursor.execute(
                    'DELETE FROM facts WHERE subject = ? AND relation = ? AND object = ?',
                    (subject, relation, obj)
                )
                self.session_manager.databases['relational'].commit()
            
            # Update research data
            if self.session_manager.research_data and 'facts' in self.session_manager.research_data:
                for i, data_fact in enumerate(self.session_manager.research_data['facts'][:]):
                    if (data_fact.get('subject') == fact.get('subject') and
                        data_fact.get('relation') == fact.get('relation') and
                        data_fact.get('object') == fact.get('object')):
                        del self.session_manager.research_data['facts'][i]
                        break
            
            deleted_count += 1
        
        # Save the session
        self.session_manager.save_session()
        
        print(f"Deleted {deleted_count} facts from source '{source_id}'.")
        print("Use 'list_deleted_facts' to view deleted facts or 'restore_fact' to recover them.")

    def export_json(self, filename: Optional[str] = None) -> None:
        """
        Export the current session data to a JSON file using the original format.
        This will ensure compatibility with the query_atom_smallworld.py format.
        """
        if not self.session_manager.current_session or not self.session_manager.research_data:
            print("No active session to export.")
            return
        
        # Use provided filename or generate one
        if not filename:
            filename = f"export_{self.session_manager.current_session}.json"
        
        # Ensure the research_data structure matches query_atom_smallworld.py expectations
        export_data: ResearchData = {
            "main_question": self.session_manager.research_data.get("main_question", "Unknown question"),
            "conclusion": self.session_manager.research_data.get("conclusion", ""),
            "timestamp": self.session_manager.research_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "results": self.session_manager.research_data.get("results", {}),
            "facts": self.session_manager.facts,
            "bibliography": self.session_manager.research_data.get("bibliography", [])
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"Session data exported to {filename}")
        except Exception as e:
            print(f"Error exporting data: {str(e)}")

    def update_fact_source(self, fact: Fact, new_source_id: str, evidence: Dict[str, Any]) -> None:
        """Update a fact's source and evidence."""
        original_source_id = fact['evidence'][0].get('source_id', 'unknown') if 'evidence' in fact and fact['evidence'] else 'unknown'
        
        # Update the evidence with the new source info
        fact['evidence'][0] = Evidence(
            source_id=new_source_id,
            url=new_source_id,
            title=evidence.get('title'),
            text=None,
            chunk_id=None,
            search_query=None,
            content=None,
            original_source_id=original_source_id
        )
        
        print(f"  Updated fact source from '{original_source_id}' to '{new_source_id}'")
        
        # Update research data if it exists
        if self.session_manager.research_data and 'facts' in self.session_manager.research_data:
            for research_fact in self.session_manager.research_data['facts']:
                if (research_fact.get('subject') == fact.get('subject') and
                    research_fact.get('relation') == fact.get('relation') and
                    research_fact.get('object') == fact.get('object')):
                    research_fact['evidence'] = fact['evidence']
                    break
        
        self.session_manager.save_session() 

    def restore_fact(self, index: int) -> None:
        """Restore a previously deleted fact."""
        if not self.session_manager.deleted_facts:
            print("No deleted facts to restore.")
            return
        
        try:
            index = int(index) - 1  # Convert to 0-based index
            if index < 0 or index >= len(self.session_manager.deleted_facts):
                print(f"Invalid fact number. Please specify a number between 1 and {len(self.session_manager.deleted_facts)}.")
                return
            
            fact = self.session_manager.deleted_facts[index]
            
            # Check if this would create a duplicate
            if self._is_duplicate_fact(fact.get('subject', ''), fact.get('relation', ''), fact.get('object', '')):
                print(f"Cannot restore fact as it would create a duplicate: {fact.get('subject', '')} {fact.get('relation', '')} {fact.get('object', '')}")
                return
            
            # Remove from deleted facts
            del self.session_manager.deleted_facts[index]
            
            # Add back to active facts
            self.session_manager.facts.append(fact)
            
            # Update the fact index
            self._fact_index.add(self._fact_to_key(fact))
            
            # Add back to database if it exists
            if self.session_manager.databases:
                try:
                    cursor = self.session_manager.databases['relational'].cursor()
                    cursor.execute(
                        'INSERT INTO facts (subject, relation, object, source_id) VALUES (?, ?, ?, ?)',
                        (fact.get('subject', ''), fact.get('relation', ''), fact.get('object', ''),
                         fact['evidence'][0].get('source_id', 'unknown') if fact.get('evidence') else 'unknown')
                    )
                    self.session_manager.databases['relational'].commit()
                except Exception as e:
                    print(f"Warning: Could not add fact back to database: {e}")
            
            # Update research data
            if self.session_manager.research_data and 'facts' in self.session_manager.research_data:
                self.session_manager.research_data['facts'].append(fact)
            
            # Save the session
            self.session_manager.save_session()
            
            print(f"Restored fact: {fact.get('subject', '')} {fact.get('relation', '')} {fact.get('object', '')}")
            
        except ValueError:
            print("Please provide a valid fact number.")

    def add_verified_fact(self, fact: Dict[str, Any]) -> bool:
        """
        Add a new fact discovered during verification.
        Returns True if fact was added, False if it was a duplicate.
        """
        # Check for duplicates using the helper method
        if self._is_duplicate_fact(fact.get('subject', ''), fact.get('relation', ''), fact.get('object', '')):
            return False
            
        # Add to facts list
        self.session_manager.facts.append(fact)
        
        # Update the fact index
        self._fact_index.add(self._fact_to_key(fact))
        
        # Add to research data
        if self.session_manager.research_data and 'facts' in self.session_manager.research_data:
            self.session_manager.research_data['facts'].append(fact)
            
        return True

    def load_session(self, session_id: str) -> None:
        """Load a research session and rebuild the fact index."""
        self.session_manager.load_session(session_id)
        # Rebuild the fact index after loading session
        self._rebuild_fact_index() 