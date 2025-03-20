from typing import TypedDict, List, Optional, Dict, Any, Union
from datetime import datetime

class Evidence(TypedDict):
    source_id: str
    url: Optional[str]
    title: Optional[str]
    text: Optional[str]
    chunk_id: Optional[str]
    search_query: Optional[str]
    content: Optional[str]
    original_source_id: Optional[str]

class Verification(TypedDict):
    status: str  # 'confirmed' | 'unverified' | 'false' | 'controversial'
    confirming: Optional[int]
    contradicting: Optional[int]
    evidence: Optional[List[Dict[str, Any]]]
    original_source_id: Optional[str]
    error: Optional[str]

class Fact(TypedDict):
    subject: str
    relation: str
    object: str
    evidence: List[Evidence]
    verification: Optional[Verification]
    created_at: datetime

class Source(TypedDict):
    id: str
    title: str
    url: Optional[str]
    content: str
    type: str
    created_at: datetime

class ResearchData(TypedDict):
    main_question: str
    conclusion: Optional[str]
    timestamp: datetime
    facts: List[Fact]
    search_metadata: Dict[str, Any]
    bibliography: List[Source] 