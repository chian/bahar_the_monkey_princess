from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import uuid
import json


@dataclass
class ThoughtNode:
    """Represents a single node in a thought chain/tree."""
    id: str
    text: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to a dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtNode':
        """Create node from a dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            metadata=data.get("metadata", {})
        )


class ThoughtChain:
    """Manages a collection of thought nodes and their relationships."""
    
    def __init__(self, root_text: Optional[str] = None):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        
        if root_text is not None:
            self.root_id = self.add_node(None, root_text)
    
    def add_node(self, parent_id: Optional[str], text: str, node_id: Optional[str] = None) -> str:
        """
        Add a new node as a child of the specified parent.
        
        Args:
            parent_id: ID of the parent node, or None for a root node
            text: Content of the thought
            node_id: Optional ID to use (if None, a UUID will be generated)
            
        Returns:
            The ID of the newly created node
        """
        # Generate ID if not provided
        if node_id is None:
            node_id = str(uuid.uuid4())
            
        # Create the new node
        node = ThoughtNode(
            id=node_id,
            text=text,
            parent_id=parent_id
        )
        
        # Add to nodes dictionary
        self.nodes[node_id] = node
        
        # If this is the first node, set it as the root
        if self.root_id is None and parent_id is None:
            self.root_id = node_id
            
        # Update parent's children list if parent exists
        if parent_id is not None:
            if parent_id not in self.nodes:
                raise ValueError(f"Parent node with ID {parent_id} does not exist")
            
            parent = self.nodes[parent_id]
            parent.children_ids.append(node_id)
            
        return node_id
    
    def edit_node(self, node_id: str, new_text: str) -> None:
        """
        Edit the text of an existing node.
        
        Args:
            node_id: ID of the node to edit
            new_text: New content for the node
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist")
            
        self.nodes[node_id].text = new_text
    
    def move_node(self, node_id: str, new_parent_id: Optional[str]) -> None:
        """
        Move a node (and its subtree) to a new parent.
        
        Args:
            node_id: ID of the node to move
            new_parent_id: ID of the new parent, or None to make it a root
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist")
            
        if new_parent_id is not None and new_parent_id not in self.nodes:
            raise ValueError(f"New parent node with ID {new_parent_id} does not exist")
            
        # Check if we're trying to make a node its own descendant
        if new_parent_id is not None:
            current = new_parent_id
            while current is not None:
                if current == node_id:
                    raise ValueError("Cannot make a node its own descendant")
                current = self.nodes[current].parent_id
        
        node = self.nodes[node_id]
        old_parent_id = node.parent_id
        
        # Remove from old parent's children
        if old_parent_id is not None and old_parent_id in self.nodes:
            self.nodes[old_parent_id].children_ids.remove(node_id)
            
        # Update the node's parent_id
        node.parent_id = new_parent_id
        
        # Add to new parent's children
        if new_parent_id is not None:
            self.nodes[new_parent_id].children_ids.append(node_id)
    
    def delete_node(self, node_id: str, delete_subtree: bool = True) -> None:
        """
        Delete a node and optionally its subtree.
        
        Args:
            node_id: ID of the node to delete
            delete_subtree: If True, delete all descendants; if False, re-parent them to the deleted node's parent
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist")
            
        if node_id == self.root_id and len(self.nodes) > 1:
            # Special handling for deleting the root
            raise ValueError("Cannot delete the root node unless it's the only node")
            
        node = self.nodes[node_id]
        
        # Remove from parent's children
        if node.parent_id is not None:
            self.nodes[node.parent_id].children_ids.remove(node_id)
        
        if delete_subtree:
            # Recursively delete all descendants
            children_to_delete = node.children_ids.copy()
            for child_id in children_to_delete:
                self.delete_node(child_id, delete_subtree=True)
        else:
            # Re-parent children to the deleted node's parent
            for child_id in node.children_ids:
                self.move_node(child_id, node.parent_id)
        
        # Remove the node itself
        del self.nodes[node_id]
        
        # Update root_id if necessary
        if node_id == self.root_id:
            self.root_id = None
    
    def get_chain(self, node_id: str) -> List[ThoughtNode]:
        """
        Get the path from root to the specified node.
        
        Args:
            node_id: ID of the target node
            
        Returns:
            List of nodes from root to the target node (inclusive)
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist")
            
        path = []
        current_id = node_id
        
        # Build the path from node to root
        while current_id is not None:
            path.append(self.nodes[current_id])
            current_id = self.nodes[current_id].parent_id
            
        # Reverse to get root-to-node order
        path.reverse()
        return path
    
    def get_subtree(self, node_id: Optional[str] = None) -> Dict[str, ThoughtNode]:
        """
        Get the subtree rooted at the specified node (or the entire tree if node_id is None).
        
        Args:
            node_id: ID of the root of the subtree, or None for the entire tree
            
        Returns:
            Dictionary mapping node IDs to nodes in the subtree
        """
        if node_id is None:
            node_id = self.root_id
            
        if node_id is None:
            return {}  # Empty tree
            
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} does not exist")
            
        # BFS to collect all nodes in the subtree
        subtree = {}
        queue = [node_id]
        
        while queue:
            current_id = queue.pop(0)
            subtree[current_id] = self.nodes[current_id]
            queue.extend(self.nodes[current_id].children_ids)
            
        return subtree
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the thought chain to a dictionary.
        
        Returns:
            Dictionary representation of the thought chain
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_id": self.root_id
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the thought chain to a JSON string.
        
        Args:
            indent: Number of spaces for indentation in the JSON string
            
        Returns:
            JSON string representation of the thought chain
        """
        return json.dumps(self.serialize(), indent=indent)
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ThoughtChain':
        """
        Create a thought chain from a serialized dictionary.
        
        Args:
            data: Dictionary representation of a thought chain
            
        Returns:
            ThoughtChain instance
        """
        chain = cls()
        chain.root_id = data.get("root_id")
        
        # Deserialize nodes
        for node_data in data.get("nodes", {}).values():
            node = ThoughtNode.from_dict(node_data)
            chain.nodes[node.id] = node
            
        return chain
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ThoughtChain':
        """
        Create a thought chain from a JSON string.
        
        Args:
            json_str: JSON string representation of a thought chain
            
        Returns:
            ThoughtChain instance
        """
        data = json.loads(json_str)
        return cls.deserialize(data)


def extract_steps_from_text(text: str, delimiter: str = "\n", parent_id: Optional[str] = None) -> ThoughtChain:
    """
    Split a block of text into steps and create a ThoughtChain.
    
    Args:
        text: The text to split into steps
        delimiter: The delimiter to use for splitting (default: newline)
        parent_id: Optional ID of a parent node to put all steps under
        
    Returns:
        ThoughtChain containing the extracted steps
    """
    chain = ThoughtChain()
    
    # If a parent_id is specified but doesn't exist yet, create it
    if parent_id is not None and parent_id not in chain.nodes:
        chain.add_node(None, "Root", node_id=parent_id)
    
    # Split the text and create nodes
    steps = [step.strip() for step in text.split(delimiter) if step.strip()]
    
    for step in steps:
        chain.add_node(parent_id, step)
    
    return chain


def build_thought_chain(nodes: Union[List[ThoughtNode], Dict[str, ThoughtNode]], 
                        parent_child_relationships: Optional[Dict[str, List[str]]] = None, 
                        root_id: Optional[str] = None) -> ThoughtChain:
    """
    Build a ThoughtChain from existing ThoughtNode objects.
    
    Args:
        nodes: List of ThoughtNode objects or dict mapping node_id to ThoughtNode
        parent_child_relationships: Optional dict mapping parent_id to list of child_ids
        root_id: Optional ID of the root node (if None, will try to determine automatically)
        
    Returns:
        A constructed ThoughtChain object
    """
    chain = ThoughtChain()
    
    # Convert list to dictionary if needed
    if isinstance(nodes, list):
        nodes_dict = {node.id: node for node in nodes}
    else:
        nodes_dict = nodes
    
    # Add all nodes to the chain
    chain.nodes = nodes_dict
    
    # Determine root if not provided
    if root_id is None:
        # Find nodes with no parent
        potential_roots = [node_id for node_id, node in nodes_dict.items() 
                           if node.parent_id is None]
        if len(potential_roots) == 1:
            chain.root_id = potential_roots[0]
        elif len(potential_roots) > 1:
            # Multiple roots found - create a new root node
            root_id = str(uuid.uuid4())
            root_node = ThoughtNode(id=root_id, text="Root", parent_id=None)
            chain.nodes[root_id] = root_node
            chain.root_id = root_id
            
            # Connect all previous roots to the new root
            for node_id in potential_roots:
                chain.nodes[node_id].parent_id = root_id
                if root_id not in chain.nodes[root_id].children_ids:
                    chain.nodes[root_id].children_ids.append(node_id)
    else:
        chain.root_id = root_id
    
    # Apply explicit parent-child relationships if provided
    if parent_child_relationships:
        for parent_id, child_ids in parent_child_relationships.items():
            if parent_id in chain.nodes:
                # Ensure children_ids is initialized
                if not hasattr(chain.nodes[parent_id], 'children_ids'):
                    chain.nodes[parent_id].children_ids = []
                
                # Add child relationships
                for child_id in child_ids:
                    if child_id in chain.nodes:
                        # Update child's parent
                        chain.nodes[child_id].parent_id = parent_id
                        
                        # Update parent's children list if not already there
                        if child_id not in chain.nodes[parent_id].children_ids:
                            chain.nodes[parent_id].children_ids.append(child_id)
    
    # Ensure consistency of parent-child relationships
    for node_id, node in chain.nodes.items():
        # Ensure each node with a parent is in that parent's children list
        if node.parent_id and node.parent_id in chain.nodes:
            if node_id not in chain.nodes[node.parent_id].children_ids:
                chain.nodes[node.parent_id].children_ids.append(node_id)
    
    return chain 


def thoughts_text_to_nodes(thoughts_text: str) -> list:
    """
    Convert a string of thoughts separated by double newlines into a flat list of ThoughtNode objects.
    Each node will have a unique id, text, parent_id, and children_ids (for a linear chain).
    The first node will have parent_id=None, each subsequent node's parent_id is the previous node's id.
    """
    steps = [step.strip() for step in thoughts_text.split("\n\n") if step.strip()]
    nodes = []
    prev_id = None
    for i, step in enumerate(steps):
        node_id = str(uuid.uuid4())
        node = ThoughtNode(
            id=node_id,
            text=step,
            parent_id=prev_id,
            children_ids=[],
            metadata={}
        )
        if prev_id is not None:
            # Set the previous node's children_ids to include this node
            nodes[-1].children_ids.append(node_id)
        nodes.append(node)
        prev_id = node_id
    return nodes 


def thoughtchain_to_text(nodes_dict: Dict[str, Any], root_id: str) -> str:
    """
    Convert a thought chain back to text format by following the linear chain from root to end.
    
    Args:
        nodes_dict: Dictionary mapping node IDs to node dictionaries
        root_id: ID of the root node to start from
        
    Returns:
        String with thought texts joined by double newlines
    """
    if root_id not in nodes_dict:
        return ""
    
    # Follow the linear chain from root to end
    texts = []
    current_id = root_id
    
    while current_id is not None:
        node = nodes_dict[current_id]
        texts.append(node['text'])
        
        # Move to the first (and should be only) child in a linear chain
        children = node.get('children_ids', [])
        if len(children) == 1:
            current_id = children[0]
        elif len(children) == 0:
            # End of chain
            current_id = None
        else:
            # Multiple children - this shouldn't happen in a linear chain, but take the first
            current_id = children[0]
    
    return "\n\n".join(texts) 