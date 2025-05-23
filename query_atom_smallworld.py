import sys
import openai
import requests
import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import HTTPError
import signal
from dotenv import load_dotenv
from thought_chain import thoughts_text_to_nodes, build_thought_chain

# Load environment variables
load_dotenv()

# Add firecrawl-py import
from firecrawl import FirecrawlApp

##############################################################################
# Configuration: Define default and agent-specific model information
##############################################################################

# Default configuration for direct OpenAI calls
DEFAULT_CONFIG = {
    "api_key": os.getenv('OPENAI_API_KEY'),
    "endpoint": "https://api.openai.com/v1",
    "model_id": "o4-mini",
    "tools": []
}

R1_CONFIG = {
    "api_key": os.getenv('TOGETHER_API_KEY'),
    "endpoint": "https://api.together.xyz/v1",
    "model_id": "deepseek-ai/DeepSeek-R1",
    "tools": []
}

# Firecrawl search API configuration
FIRECRAWL_CONFIG = {
    "api_key": os.getenv('FIRECRAWL_API_KEY')
}

# Agent-specific configurations
AGENT_CONFIGS = {
    "supervisor": {
        **DEFAULT_CONFIG,
        #"model_id": "gpt-4.5-preview",
    },
    "query_generation": {  # Instead of "generation"
        **DEFAULT_CONFIG,
    },
    "information_extraction": {  # Instead of "reflection"
        **R1_CONFIG,
        "max_chunk_size": 120000,
        "max_chunks": 30
    },
    "evidence_evaluation": {  # Instead of "ranking"
        **DEFAULT_CONFIG,
    },
    "answer_synthesis": {  # Instead of "evolution"
        **DEFAULT_CONFIG,
    },
    "reasoning_verification": {  # Instead of "proximity_check"
        **DEFAULT_CONFIG,
    },
    "fact_atomization": {
        **DEFAULT_CONFIG,
        #"model_id": "gpt-4.5-preview",
        "max_chunk_size": 120000,
        "max_chunks": 30
    },
    "conclusion_formation": {  # Instead of "meta_review"
        **R1_CONFIG,
    },
    "reasoning_continuation": {
        **R1_CONFIG,
        "model_id": "deepseek-ai/DeepSeek-V3"
    },
}

##############################################################################
# Scientific Relations Dictionary for Fact Atomization
##############################################################################

# Revised dictionary of explicit, factual relationships (plus a fallback)
SCIENTIFIC_RELATIONS = {
    "causes":          "X causes Y",
    "increases":       "X increases Y",
    "reduces":         "X reduces Y",
    "equals":          "X equals Y",
    "property_of":     "X is a property of Y",
    "associated_with": "X is associated with Y",
    "binds_to":        "X physically binds to Y",
    "inhibits":        "X inhibits Y",
    "activates":       "X activates Y",
    "exacerbates":     "X exacerbates Y (worsens/amplifies)",
    "predates_on":     "X preys on Y",
    "competes_with":   "X competes with Y",
    "symbiotic_with":  "X has a symbiotic relationship with Y",
    "pollinates":      "X pollinates Y",
    "is_habitat_for":  "X is a habitat for Y",
    "hosts":           "X hosts Y (parasite/pathogen)",
    "descended_from":       "X descended from Y evolutionarily",
    "common_ancestor_with": "X has a common ancestor with Y",
    "diverged_from":        "X diverged from Y in evolution",
    "hybridizes_with":      "X can hybridize with Y",
    "unknown_relationship": "Fallback if no known relation applies"
}

##############################################################################
# Agent Prompts
##############################################################################

def get_agent_config_and_prompt(agent_type: str) -> tuple[Dict[str, Any], str]:
    """
    Returns the configuration and prompt for a specific agent type.
    """
    config = AGENT_CONFIGS.get(agent_type, DEFAULT_CONFIG)
    
    if agent_type == "supervisor":
        prompt = (
            "You are the Supervisor Agent in a scientific question answering system. "
            "You coordinate a series of specialized agents to comprehensively answer "
            "scientific questions through research, reasoning, and evidence evaluation. "
            "You identify knowledge gaps that require follow-up investigation, manage "
            "the workflow between agents, and ensure the final answer addresses the "
            "original question with proper scientific reasoning."
        )
    elif agent_type == "query_generation":
        prompt = (
            "You are an expert at generating search queries for scientific research. "
            "Your task is to convert questions into effective search queries that will "
            "yield relevant scientific information."
        )
    elif agent_type == "information_extraction":
        prompt = (
            "You are the Information Extraction Agent in a scientific question answering system. "
            "Analyze search results and extract key information relevant to the scientific question. "
            "Focus on identifying facts, relationships, mechanisms, evidence, research findings, "
            "and scientific consensus. Organize the information logically and note conflicting "
            "evidence when present. Also identify important entities, concepts, and how they relate."
        )
    elif agent_type == "evidence_evaluation":
        prompt = (
            "You are the Evidence Evaluation Agent in a scientific question answering system. "
            "Critically assess the extracted information for quality, reliability, and relevance. "
            "Evaluate the strength of evidence using scientific criteria: study design, sample size, "
            "methods, peer review status, recency, consensus agreement, and potential biases. "
            "Clearly distinguish between established facts, emerging evidence, and speculation. "
            "Rate confidence levels for key claims and identify remaining uncertainties."
        )
    elif agent_type == "answer_synthesis":
        prompt = (
            "You are the Answer Synthesis Agent in a scientific question answering system. "
            "Synthesize the evaluated evidence into a comprehensive answer to the scientific question. "
            "Present a clear line of reasoning, connecting evidence to conclusions. Address multiple "
            "perspectives and levels of explanation (e.g., molecular, cellular, systemic) when appropriate. "
            "Acknowledge limitations and uncertainties in the current understanding. Structure your "
            "answer to progressively build understanding of complex topics."
        )
    elif agent_type == "reasoning_verification":
        prompt = (
            "You are the Reasoning Verification Agent in a scientific question answering system. "
            "Verify the logical structure and scientific accuracy of synthesized answers. "
            "Check for reasoning errors, such as correlation/causation fallacies, "
            "inappropriate generalizations, or logical inconsistencies. Ensure that conclusions "
            "follow from evidence and that all key claims are supported. Identify gaps in the "
            "reasoning chain that need to be addressed for a complete scientific explanation."
        )
    elif agent_type == "fact_atomization":
        prompt = (
            "You are the Fact Atomization Agent in a scientific question answering system. "
            "Extract factual relationships from scientific text as Subject-Relation-Object triples. "
            "Use only the predefined set of relations. For each excerpt, identify as many scientific "
            "facts as possible, focusing on causal, functional, structural, and evolutionary "
            "relationships. If no specific relation applies, use 'unknown_relationship'."
            "\n\nIMPORTANT: Be extremely careful with tabular data (content with columns and rows). "
            "When processing tables:"
            "\n1. Identify the column headers and understand what each column represents"
            "\n2. Extract facts ONLY when you can clearly connect values across columns with correct meaning"
            "\n3. Never connect values from different rows as if they were related"
            "\n4. If you see 'TABLE_DATA:' markers, treat this content with extra caution"
            "\n5. Do NOT extract facts when relationships between columns are ambiguous"
            "\nFormat output as structured JSON for downstream processing."
        )
    elif agent_type == "conclusion_formation":
        prompt = (
            "You are the Conclusion Formation Agent in a scientific question answering system. "
            "Synthesize all findings from the research process into a final conclusion that "
            "directly answers the original scientific question. Present a clear, coherent reasoning "
            "chain connecting evidence to conclusions. Highlight major supporting evidence, acknowledge "
            "uncertainties, and explain the current scientific consensus. Include practical implications "
            "and future research directions where appropriate. Your conclusion should be accessible "
            "to an educated non-specialist while maintaining scientific accuracy."
        )
    else:
        prompt = (
            f"You are the {agent_type} Agent in a scientific question answering system. "
            "Provide expert assistance based on your specialized role."
        )
    
    return config, prompt

##############################################################################
# Search with Firecrawl integration (using official package)
##############################################################################

# Initialize the Firecrawl client
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_CONFIG["api_key"])

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Search timed out")

def execute_search(query: str, num_results: int = 5, use_timeout_signal: bool = True) -> List[Dict[str, Any]]:
    """
    Execute a search query using Firecrawl
    Args:
        query: The search query
        num_results: Number of results to return
        use_timeout_signal: Whether to use signal-based timeout (only works in main thread)
    Returns:
        List of search results with content
    """
    print(f"Searching with Firecrawl: {query}")
    results = []
    
    try:
        # Only use signal-based timeout in the main thread
        if use_timeout_signal:
            # Set timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
        
        # Execute search with Firecrawl - use EXACT params from v3
        search_response = firecrawl_app.search(
            query=query, 
            params={
                'timeout': 30000,
                'limit': num_results,
                'scrapeOptions': {'formats': ['markdown']}
            }
        )
        
        # Reset alarm if we used it
        if use_timeout_signal:
            signal.alarm(0)
        
        # Get data items based on the response structure
        if hasattr(search_response, 'data'):
            data_items = search_response.data
        elif isinstance(search_response, dict) and 'data' in search_response:
            data_items = search_response['data']
        else:
            print(f"Unexpected response format: {type(search_response)}")
            data_items = []
            
        # Check if we have results using data_items
        if not search_response or len(data_items) == 0:
            print(f"No results found for query: {query}")
            return []
        
        # Process results, similar to original function
        formatted_results = []
        
        # Handle different response structures exactly as in v1
        for item in data_items:
            # Extract URL - match deep-research.ts access pattern
            url = item.url if hasattr(item, 'url') else item.get('url', '')
            
            # Extract markdown content - match deep-research.ts access pattern
            content = item.markdown if hasattr(item, 'markdown') else item.get('markdown', '')
            if not content:
                content = item.content if hasattr(item, 'content') else item.get('content', '')
            
            # Get title
            title = item.title if hasattr(item, 'title') else item.get('title', url)
            
            formatted_results.append({
                "title": title,
                "url": url,
                "source": url.split("//")[-1].split("/")[0] if "//" in url else "unknown",
                "snippet": content[:500] + "..." if len(content) > 500 else content,
                "content": content,
                "query": query
            })
        
        # Only print response dictionary if zero results found (as in v1)
        if len(formatted_results) == 0:
            # Convert response to dictionary format
            if hasattr(search_response, '__dict__'):
                response_dict = search_response.__dict__
            elif isinstance(search_response, dict):
                response_dict = search_response
            else:
                response_dict = {"response": str(search_response)}
                
            print(f"Zero results - Response dictionary: {response_dict}")
        
        print(f"Found {len(formatted_results)} results from Firecrawl")
        
        # Track visited URLs globally
        global visited_urls
        visited_urls.extend([item["url"] for item in formatted_results if item["url"]])
        
        return formatted_results
        
    except TimeoutError:
        print(f"Search timed out after 30 seconds: {query}")
        return []
    except Exception as e:
        print(f"Error searching with Firecrawl: {e}")
        return []

# Global list to track visited URLs for report generation
visited_urls = []


def execute_parallel_searches(queries: List[str], num_results_per_query: int = 5) -> List[Dict[str, Any]]:
    """
    Execute multiple searches in parallel with a concurrency limit
    """
    CONCURRENCY_LIMIT = 2  # Match deep-research.ts ConcurrencyLimit
    
    def search_worker(query):
        # Disable signals when running in worker threads
        return execute_search(query, num_results_per_query, use_timeout_signal=False)
    
    all_results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        results = list(executor.map(search_worker, queries))
        for query_results in results:
            all_results.extend(query_results)
    
    return all_results

##############################################################################
# Helper function to call an agent using direct API calls
##############################################################################

def call_agent(
    agent_type: str,
    user_prompt: str,
    additional_context: str = "",
    override_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call an agent using its configuration and prompt via direct API calls.
    """
    # Get agent configuration and prompt
    config, system_prompt = get_agent_config_and_prompt(agent_type)
    
    # Apply any configuration overrides
    if override_config:
        for key, value in override_config.items():
            config[key] = value
    
    # Prepare the system + user messages
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if additional_context:
        messages.append({"role": "assistant", "content": additional_context})

    messages.append({"role": "user", "content": user_prompt})
    
    # Prepare API parameters - without any token limits
    api_params = {
        "model": config["model_id"],
        "messages": messages,
    }
    
    # Pass tools if relevant in your code
    tools = config.get("tools", [])
    if tools:
        api_params["tools"] = tools
    
    # Create the openai client
    client = openai.OpenAI(
        api_key=config["api_key"], 
        base_url=config["endpoint"]
    )
    
    response = client.chat.completions.create(**api_params)
    
    # Handle potential tool calls (if your model uses function calling)
    if (
        hasattr(response.choices[0].message, 'tool_calls') and 
        response.choices[0].message.tool_calls
    ):
        tool_results = []
        for tool_call in response.choices[0].message.tool_calls:
            try:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)
                tool_result = execute_tool(tool_name, tool_input)
                tool_results.append(f"Tool: {tool_name}\nResult: {tool_result}")
            except Exception as e:
                tool_results.append(f"Error executing tool {tool_call.function.name}: {str(e)}")
        
        # Follow up with the tool results
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        messages.append({"role": "user", "content": "Tool results:\n" + "\n\n".join(tool_results)})
        
        # Make a second call without token limits
        follow_up_response = client.chat.completions.create(**api_params)
        return follow_up_response.choices[0].message.content
    
    return response.choices[0].message.content

##############################################################################
# Fact Atomization Function
##############################################################################

def detect_and_process_tables(text: str) -> str:
    """
    Detect potential tabular data and mark it explicitly, 
    while leaving non-tabular text unchanged. Returns the full text
    with only the table portions modified with special markers.
    """
    # Quick check for common table indicators before doing full processing
    table_indicators = ['\t', '  |', '| ', ' |  ']
    has_potential_tables = any(indicator in text for indicator in table_indicators)
    
    # If no potential tables detected, return the original text unchanged
    if not has_potential_tables and text.count('|') < 10:
        return text  # Return ALL the original text
    
    lines = text.split('\n')
    processed_lines = lines.copy()  # Create a copy to modify
    
    # Track potential table line ranges
    table_ranges = []
    current_table_start = None
    consecutive_table_lines = 0
    
    # First pass: identify potential table ranges
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            if current_table_start is not None and consecutive_table_lines >= 3:
                # We found a table that ended
                table_ranges.append((current_table_start, i-1))
            current_table_start = None
            consecutive_table_lines = 0
            continue
            
        # Look for common table indicators: multiple spaces or tabs that align columns
        if ('\t' in line) or ('  ' in line and '|' in line) or line.count('|') >= 2:
            if current_table_start is None:
                current_table_start = i
            consecutive_table_lines += 1
        else:
            # This is regular text - it stays unchanged
            if current_table_start is not None and consecutive_table_lines >= 3:
                # We found a table that ended
                table_ranges.append((current_table_start, i-1))
            current_table_start = None
            consecutive_table_lines = 0
    
    # Handle case where table is at the end of text
    if current_table_start is not None and consecutive_table_lines >= 3:
        table_ranges.append((current_table_start, len(lines)-1))
    
    # If no tables found after detailed scan, return original text
    if not table_ranges:
        return text  # Return ALL the original text
    
    # Second pass: only modify the identified table ranges
    # All other lines remain unchanged
    for start, end in table_ranges:
        # Add table start marker
        processed_lines[start] = "TABLE_DATA_START:\n" + processed_lines[start]
        
        # Add table end marker
        processed_lines[end] += "\nTABLE_DATA_END"
        
        # Mark each line in between as table data
        for i in range(start+1, end):
            if processed_lines[i].strip():  # Skip empty lines
                processed_lines[i] = "TABLE_ROW: " + processed_lines[i]
    
    # Recombine the text - this includes ALL text (table and non-table)
    processed_text = '\n'.join(processed_lines)
    
    # Log information about tables found
    print(f"Detected {len(table_ranges)} tables in the text")
    
    return processed_text  # Return the FULL text with only table sections modified

def parse_json_response(response: str) -> Union[List, Dict, None]:
    """
    Safely parse a JSON response from an agent, handling various formats and errors.
    
    Args:
        response: String response potentially containing JSON
        
    Returns:
        Parsed JSON object (list or dict) or None if parsing fails
    """
    # First try to find JSON between triple backticks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, response)
    
    if matches:
        # Try each matched content
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON found in code blocks, look for JSON anywhere in the response
    try:
        # Try to find JSON-like content with { } or [ ]
        curly_match = re.search(r"\{[\s\S]*\}", response)
        if curly_match:
            return json.loads(curly_match.group(0))
        
        square_match = re.search(r"\[[\s\S]*\]", response)
        if square_match:
            return json.loads(square_match.group(0))
    except json.JSONDecodeError:
        pass
    
    # Last resort: try parsing the whole response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        # If all parsing attempts fail, return None
        return None

def atomize_facts(text: str, source_info: Dict = None) -> List[Dict[str, str]]:
    """
    Extract factual relationships from the given text using the fact atomization agent.
    Returns a list of dicts with fact details and source information.
    """
    # Get chunk parameters from agent config
    agent_config = AGENT_CONFIGS["fact_atomization"]
    max_chunk_size = agent_config.get("max_chunk_size", 120000)  # Default if not in config
    max_chunks = agent_config.get("max_chunks", 30)  # Default if not in config
    
    # Check text length - if too long, process in chunks
    if len(text) > max_chunk_size:
        print(f"Text too long ({len(text)} chars), processing in chunks...")
        
        # Process text in chunks and combine results
        chunks = break_text_into_chunks(text, max_chunk_size)
        all_facts = []
        
        # Only process up to max_chunks
        for i, chunk in enumerate(chunks[:max_chunks]):
            print(f"Processing chunk {i+1}/{min(len(chunks), max_chunks)}...")
            # Create chunk-specific source info
            chunk_source_info = source_info.copy() if source_info else {}
            if chunk_source_info:
                chunk_source_info["chunk_id"] = f"{chunk_source_info.get('chunk_id', '0')}.{i+1}"
            
            # Process this chunk
            chunk_facts = atomize_facts(chunk, chunk_source_info)
            all_facts.extend(chunk_facts)
            
        if len(chunks) > max_chunks:
            print(f"Note: Text was too long. Only processed first {max_chunks} chunks.")
        
        return all_facts
    
    # First, detect and process any tables in the text
    processed_text = detect_and_process_tables(text)
    
    # Use your original prompt format that you worked hard to create
    fact_prompt = f"""
You are a system that extracts factual biology/ecology/evolution relationships
from text as a JSON array of Subject–Relation–Object triples.

Use ONLY the following relations, or 'unknown_relationship' if uncertain:
{json.dumps(list(SCIENTIFIC_RELATIONS.keys()), indent=2)}

Format for each fact:
[
  {{
    "subject": "...",
    "relation": "...",
    "object": "..."
  }},
  ...
]

Follow these examples:

EXAMPLE 1 (knockout or mutation context):
Text: "Knocking out Gene A abolishes Trait B. Gene A is related to metabolic pathways."
JSON facts:
[
  {{
    "subject": "Gene A",
    "relation": "essential_to",
    "object": "Trait B"
  }},
  {{
    "subject": "Gene A",
    "relation": "associated_with",
    "object": "metabolic pathways"
  }}
]

EXAMPLE 2 (predator-prey, unknown fallback):
Text: "Owls predate on mice. Owls do something weird with trees (not sure what)."
JSON facts:
[
  {{
    "subject": "Owls",
    "relation": "predates_on",
    "object": "mice"
  }},
  {{
    "subject": "Owls",
    "relation": "unknown_relationship",
    "object": "trees"
  }}
]

EXAMPLE 3 (molecular or general biology):
Text: "Protein X binds to Receptor Y. Drug Z inhibits Protein X."
JSON facts:
[
  {{
    "subject": "Protein X",
    "relation": "binds_to",
    "object": "Receptor Y"
  }},
  {{
    "subject": "Drug Z",
    "relation": "inhibits",
    "object": "Protein X"
  }}
]

EXAMPLE 4 (table data):
Text: "TABLE_DATA_START:
Species | Average Lifespan (years) | Conservation Status
African Elephant | 60-70 | Vulnerable
Blue Whale | 80-90 | Endangered
Giant Panda | 20-30 | Vulnerable
TABLE_DATA_END"
JSON facts:
[
  {{
    "subject": "African Elephant",
    "relation": "property_of",
    "object": "average lifespan of 60-70 years"
  }},
  {{
    "subject": "African Elephant",
    "relation": "property_of",
    "object": "Vulnerable conservation status"
  }},
  {{
    "subject": "Blue Whale",
    "relation": "property_of",
    "object": "average lifespan of 80-90 years"
  }},
  {{
    "subject": "Blue Whale",
    "relation": "property_of",
    "object": "Endangered conservation status"
  }},
  {{
    "subject": "Giant Panda",
    "relation": "property_of",
    "object": "average lifespan of 20-30 years"
  }},
  {{
    "subject": "Giant Panda",
    "relation": "property_of",
    "object": "Vulnerable conservation status"
  }}
]

---
Now apply the same approach to this text:

\"\"\"{processed_text}\"\"\"
Output your facts as a JSON array.
If no known relation applies, use 'unknown_relationship'.
"""
    
    # Call the fact atomization agent
    atomization_output = call_agent(
        agent_type="fact_atomization",
        user_prompt=fact_prompt
    )
    
    # Parse the JSON from the agent's response
    final_facts = []
    try:
        # Find JSON content within the response if it's wrapped in text
        json_start = atomization_output.find('[')
        json_end = atomization_output.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_content = atomization_output[json_start:json_end]
            extracted_facts = json.loads(json_content)
        else:
            # Try to parse the whole response as JSON
            extracted_facts = json.loads(atomization_output)
            
        if not isinstance(extracted_facts, list):
            print("Extracted facts is not a list")
            return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw output: {atomization_output[:100]}...")
        
        # Attempt to salvage response using parse_json_response
        extracted_facts = parse_json_response(atomization_output)
        if not extracted_facts:
            print("Failed to parse response as JSON")
            return []
    
    # Validate each triple, ensuring the relation is in our set
    for item in extracted_facts:
        if (
            isinstance(item, dict) and
            "subject" in item and
            "relation" in item and
            "object" in item
        ):
            subject = item["subject"].strip()
            relation = item["relation"].strip()
            obj = item["object"].strip()
            
            # Skip empty fields
            if not (subject and relation and obj):
                continue
                
            # Enforce valid relation
            if relation not in SCIENTIFIC_RELATIONS:
                # Force fallback if unknown relation
                relation = "unknown_relationship"
            
            # Build triple with source information
            triple = {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "evidence": []  # List to hold multiple pieces of evidence
            }
            
            # Add source information if available
            if source_info:
                evidence = {
                    "query": source_info.get("query", "Unknown query"),
                    "source_id": source_info.get("source_id", "Unknown source"),
                    "chunk_id": source_info.get("chunk_id", "Unknown chunk"),
                    "source_url": source_info.get("source_url", ""),
                    "source_text": source_info.get("text", ""),
                    "investigation_type": source_info.get("investigation_type", "unknown")
                }
                triple["evidence"].append(evidence)
                
            final_facts.append(triple)
    
    return final_facts

def break_text_into_chunks(text: str, max_length: int) -> List[str]:
    """
    Break text into semantically meaningful chunks of approximately max_length characters.
    Try to break at paragraph boundaries when possible.
    """
    if len(text) <= max_length:
        return [text]
    chunks = []
    paragraphs = text.split('\n\n')
    
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        # If this paragraph alone exceeds max length, we need to split it
        if len(paragraph) > max_length:
            # Add any accumulated paragraphs first
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split the long paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sent_chunk = []
            sent_length = 0
            
            for sentence in sentences:
                if sent_length + len(sentence) + 1 > max_length:
                    if sent_chunk:
                        chunks.append(' '.join(sent_chunk))
                    sent_chunk = [sentence]
                    sent_length = len(sentence)
                else:
                    sent_chunk.append(sentence)
                    sent_length += len(sentence) + 1  # +1 for space
            
            if sent_chunk:
                chunks.append(' '.join(sent_chunk))
        
        # Normal case - paragraph fits in a chunk
        elif current_length + len(paragraph) + 2 <= max_length:  # +2 for '\n\n'
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 2
        else:
            # Adding this paragraph would exceed the limit, so start a new chunk
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_length = len(paragraph)
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

##############################################################################
# Parser utilities for structured outputs
##############################################################################

def parse_queries(query_text: str, expected_count: int = 4) -> List[str]:
    """
    Parse search queries from the agent's output.
    """
    lines = query_text.split("\n")
    queries = []
    
    for line in lines:
        line = line.strip()
        # Look for numbered queries (1., 2., etc.)
        if line and (line[0].isdigit() and len(line) > 2 and line[1] in ['.', ')', ':']):
            query = line[2:].strip()
            queries.append(query)
        # Or bullet points
        elif line.startswith("•") or line.startswith("-") or line.startswith("*"):
            query = line[1:].strip()
            queries.append(query)
    
    # If we couldn't parse, try a simple split on newlines for non-empty lines
    if not queries:
        queries = [line.strip() for line in lines if line.strip()]
    
    # Limit to expected count
    return queries[:expected_count]

def parse_questions(questions_text: str, expected_count: int = 3) -> List[str]:
    """
    Parse follow-up questions from the agent's output.
    """
    return parse_queries(questions_text, expected_count)

def format_results_to_context(results: Dict[str, Any]) -> str:
    """
    Format the nested results structure into a flat context string.
    """
    context = f"Main Question: {results['question']}\n\n"
    context += f"Main Answer: {results['answer']}\n\n"
    context += f"Evidence: {results['evidence']}\n\n"
    
    context += "Extracted Facts:\n"
    for fact in results['facts']:
        fact_str = f"- {fact['subject']} {fact['relation']} {fact['object']}"
        
        # Add source URL if available
        if "evidence" in fact and fact["evidence"]:
            for evidence in fact["evidence"]:
                if evidence.get("source_url"):
                    fact_str += f" (source: {evidence['source_url']})"
                    break
        
        context += fact_str + "\n"
    
    if 'followups' in results:
        context += "Follow-up Investigations:\n"
        for i, followup in enumerate(results['followups']):
            context += f"\nFollow-up Question {i+1}: {followup['question']}\n"
            context += f"Answer: {followup['answer']}\n"
            context += f"Evidence: {followup['evidence']}\n"
    
    return context

def filter_relevant_evidence(
    facts: list, 
    query: str,  
    keyword_batch_size: int = 100
) -> list:
    """
    Iteratively use an LLM to select relevant keywords from the set of subjects and objects (no new keywords allowed),
    then filter facts by those keywords. Handles large keyword sets by batching.
    """
    import json, re

    # Step 1: Collect unique subjects and objects (deduplicated)
    keywords_set = set()
    for fact in facts:
        if fact.get('subject'):
            keywords_set.add(fact['subject'])
        if fact.get('object'):
            keywords_set.add(fact['object'])
    keywords_list = list(keywords_set)

    # Step 2: Iteratively ask LLM to select relevant keywords in batches
    selected_keywords = set()
    for i in range(0, len(keywords_list), keyword_batch_size):
        batch = keywords_list[i:i+keyword_batch_size]
        prompt = (
            f"Given the scientific question:\n\"{query}\"\n\n"
            f"Here is a list of possible keywords (subjects and objects):\n"
            + "\n".join(f"- {kw}" for kw in batch) +
            f"\n\nSelect ONLY from the list above the keywords or key phrases that are most relevant for filtering facts to answer the question. "
            f"Do not invent or generate new keywords. Only select from the list. "
            f"Return ONLY a JSON array of selected keywords, e.g. [\"keyword1\", \"keyword2\"]."
        )
        response = call_agent(
            agent_type="supervisor",
            user_prompt=prompt
        )
        try:
            batch_selected = json.loads(re.search(r"\[.*?\]", response, re.DOTALL).group(0))
            selected_keywords.update(batch_selected)
        except Exception as e:
            print(f"Failed to parse LLM response: {e}\nResponse: {response}")

    # Step 3: Filter facts by selected keywords
    relevant_facts = []
    for fact in facts:
        if (fact.get('subject') in selected_keywords) or (fact.get('object') in selected_keywords):
            relevant_facts.append(fact)

    return relevant_facts

##############################################################################
# Core iterative reasoning process
##############################################################################

def process_iteration(
    question: str, 
    context: List[str], 
    depth_remaining: int,
    breadth: int = 4,
    is_main_investigation: bool = True,
    session_id: str = None,
    session_manager = None
) -> Dict[str, Any]:
    """
    Process one iteration of the scientific reasoning workflow.
    If session_id and session_manager are provided, save progress incrementally.
    """
    # Get chunk parameters from information extraction agent config
    extraction_config = AGENT_CONFIGS["information_extraction"]
    max_chunk_size = extraction_config.get("max_chunk_size", 120000)
    max_chunks = extraction_config.get("max_chunks", 30)
    
    print(f"\n=== Processing Question: {question} (Depth: {depth_remaining}) ===\n")
    
    # Create a result dictionary to track progress
    current_results = {
        "question": question,
        "context": context,
        "search_queries": [],
        "facts": []
    }
    
    # 1. Generate search queries based on the question and context
    context_str = "\n".join(context) if context else ""
    
    # More closely match the original deep-research.ts prompt style
    query_prompt = (
        f"Given the following prompt from the user, generate a list of SERP queries to research the topic. "
        f"Return a maximum of {breadth} queries, but feel free to return less if the original prompt is clear. "
        f"Make sure each query is unique and not similar to each other: "
        f"<prompt>{question}</prompt>\n\n"
    )
    
    # Add previous context as "learnings" if available
    if context_str:
        query_prompt += (
            f"Here are some learnings from previous research, use them to generate more specific queries: "
            f"{context_str}"
        )
    
    queries_output = call_agent(
        agent_type="query_generation",
        user_prompt=query_prompt
    )
    parsed_queries = parse_queries(queries_output, expected_count=breadth)
    
    print("=== Generated Search Queries ===")
    for i, query in enumerate(parsed_queries):
        print(f"{i+1}. {query}")
    print("")
    
    # Update current results
    current_results["search_queries"] = parsed_queries
    
    # Save progress if session_id and session_manager provided
    if session_id and session_manager:
        session_manager.research_data["results"] = current_results
        session_manager.save_session()
        print(f"Saved search queries to session: {session_id}")
    
    # Track search queries and sources
    search_metadata = {
        "queries": parsed_queries,
        "sources": []  # Will store bibliography entries
    }
    
    # Execute searches
    search_results = []
    source_counter = 1  # For bibliography numbering
    
    for query in parsed_queries:
        results = execute_search(query)
        for result in results:
            # Add metadata to track the source
            result["query"] = query
            result["source_id"] = source_counter
            
            # Add to bibliography
            search_metadata["sources"].append({
                "id": source_counter,
                "url": result.get("url", "No URL available"),
                "title": result.get("title", "Untitled source"),
                "chunks": []  # Will store text chunks
            })
            
            source_counter += 1
            search_results.append(result)
    
    # Update current results
    current_results["search_results"] = search_results
    current_results["search_metadata"] = search_metadata
    
    # Save progress if session_id and session_manager provided
    if session_id and session_manager:
        session_manager.research_data["results"] = current_results
        session_manager.save_session()
        print(f"Saved search results to session: {session_id}")
    
    # Format search results for the agent - respect chunk limits
    search_results_str = ""
    processed_results = 0
    current_size = 0
    
    for i, result in enumerate(search_results):
        result_text = f"Result {i+1}:\n"
        result_text += f"Title: {result['title']}\n"
        result_text += f"Source: {result['source']}\n"
        result_text += f"URL: {result['url']}\n"
        result_text += f"Excerpt: {result['snippet']}\n\n"
        
        # Check if adding this result would exceed max_chunk_size
        if current_size + len(result_text) > max_chunk_size:
            if processed_results >= max_chunks:
                print(f"Note: Limited to {max_chunks} chunks of search results")
                break
            processed_results += 1
            current_size = len(result_text)
        else:
            current_size += len(result_text)
        
        search_results_str += result_text
    
    # 3. Extract information from search results
    extract_prompt = (
        f"Extract relevant scientific information from these search results to help answer "
        f"this question:\n\n{question}\n\n"
        f"Search Results:\n{search_results_str}\n\n"
        f"Organize the information by key concepts, findings, and relationships. "
        f"Note any conflicting evidence or perspectives."
    )
    
    extraction_output = call_agent(
        agent_type="information_extraction",
        user_prompt=extract_prompt
    )
    
    print("\n=== Information Extraction Results ===")
    print(extraction_output)
    print("")
    
    # Update current results
    current_results["extraction_output"] = extraction_output
    
    # Save progress if session_id and session_manager provided
    if session_id and session_manager:
        session_manager.research_data["results"] = current_results
        session_manager.save_session()
        print(f"Saved information extraction to session: {session_id}")
    
    # 4. Evaluate evidence quality
    evaluate_prompt = (
        f"Evaluate the quality, reliability, and relevance of this extracted information "
        f"for answering the scientific question:\n\n{question}\n\n"
        f"Extracted Information:\n{extraction_output}\n\n"
        f"Assess strength of evidence, identify knowledge gaps, potential biases, "
        f"and assign confidence levels to key claims."
    )
    
    evaluation_output = call_agent(
        agent_type="evidence_evaluation",
        user_prompt=evaluate_prompt
    )
    
    print("\n=== Evidence Evaluation Results ===")
    print(evaluation_output)
    print("")
    
    # Update current results
    current_results["evaluation_output"] = evaluation_output
    
    # Save progress if session_id and session_manager provided
    if session_id and session_manager:
        session_manager.research_data["results"] = current_results
        session_manager.save_session()
        print(f"Saved evidence evaluation to session: {session_id}")
    
    # 5. Synthesize an answer based on evidence
    synthesize_prompt = (
        f"Synthesize a comprehensive answer to this scientific question:\n\n{question}\n\n"
        f"Based on this evaluated evidence:\n{evaluation_output}\n\n"
        f"Present a clear scientific explanation with logical reasoning. "
        f"Address different levels of explanation when relevant and acknowledge "
        f"uncertainties or limitations in current understanding."
    )
    
    synthesis_output = call_agent(
        agent_type="answer_synthesis",
        user_prompt=synthesize_prompt
    )
    
    print("\n=== Answer Synthesis Results ===")
    print(synthesis_output)
    print("")
    
    # Update current results
    current_results["synthesis_output"] = synthesis_output
    
    # Save progress if session_id and session_manager provided
    if session_id and session_manager:
        session_manager.research_data["results"] = current_results
        session_manager.save_session()
        print(f"Saved answer synthesis to session: {session_id}")
    
    # 6. Verify reasoning
    verify_prompt = (
        f"Verify the scientific reasoning in this answer to the question:\n\n{question}\n\n"
        f"Answer to verify:\n{synthesis_output}\n\n"
        f"Check for logical fallacies, unsupported claims, correct interpretation of evidence, "
        f"and ensure conclusions follow from the presented evidence. Identify any gaps "
        f"in the reasoning chain."
    )
    
    verification_output = call_agent(
        agent_type="reasoning_verification",
        user_prompt=verify_prompt
    )
    
    print("\n=== Reasoning Verification Results ===")
    print(verification_output)
    print("")
    
    # Update current results
    current_results["verification_output"] = verification_output
    
    # Save progress if session_id and session_manager provided
    if session_id and session_manager:
        session_manager.research_data["results"] = current_results
        session_manager.save_session()
        print(f"Saved reasoning verification to session: {session_id}")
    
    # Adjust the answer based on verification feedback
    refine_prompt = (
        f"Refine your answer to this scientific question:\n\n{question}\n\n"
        f"Original answer:\n{synthesis_output}\n\n"
        f"Verification feedback:\n{verification_output}\n\n"
        f"Please address the feedback to improve the scientific accuracy and "
        f"logical coherence of your answer."
    )
    
    refined_answer = call_agent(
        agent_type="answer_synthesis",
        user_prompt=refine_prompt
    )
    
    print("\n=== Refined Answer ===")
    print(refined_answer)
    print("")
    
    # Update current results with final answer
    current_results["answer"] = refined_answer
    
    # Save progress if session_id and session_manager provided
    if session_id and session_manager:
        session_manager.research_data["results"] = current_results
        session_manager.save_session()
        print(f"Saved refined answer to session: {session_id}")
    
    # Rest of the function continues as before...
    
    # Continue with fact extraction, etc...
    
    # If we're supposed to go deeper, generate and process followup questions
    followups = []
    if depth_remaining > 1:
        followup_prompt = (
            f"Based on the answer to the original question:\n\n{question}\n\n"
            f"Current answer:\n{refined_answer}\n\n"
            f"Generate 3 targeted followup questions to investigate key aspects that would "
            f"enhance our understanding of the topic. Focus on areas that need deeper "
            f"explanation or aspects that weren't fully addressed in the current answer."
        )
        
        followup_output = call_agent(
            agent_type="query_generation",
            user_prompt=followup_prompt
        )
        
        followup_questions = parse_questions(followup_output, expected_count=3)
        print("\n=== Follow-up Questions ===")
        for i, q in enumerate(followup_questions):
            print(f"{i+1}. {q}")
        print("")
        
        # Process each followup question recursively
        for followup_question in followup_questions:
            # Add the current answer as context for the followup
            followup_context = context.copy()
            followup_context.append(refined_answer)
            
            # Recursively process the followup (with decreased depth)
            followup_result = process_iteration(
                question=followup_question,
                context=followup_context,
                depth_remaining=depth_remaining-1,
                breadth=breadth,
                is_main_investigation=False,
                session_id=session_id,
                session_manager=session_manager
            )
            
            followups.append(followup_result)
            
            # Update current results after each followup
            current_results["followups"] = followups
            
            # Save progress if session_id and session_manager provided
            if session_id and session_manager:
                session_manager.research_data["results"] = current_results
                session_manager.save_session()
                print(f"Saved followup result to session: {session_id}")
    
    # Atomize facts from the final answer
    facts = atomize_facts(refined_answer, {
        "type": "answer",
        "question": question,
        "investigation_type": "main" if is_main_investigation else "followup"
    })
    
    # Store all results in the structured format
    result = {
        "question": question,
        "context": context,
        "search_queries": parsed_queries,
        "search_metadata": search_metadata,
        "extraction_output": extraction_output,
        "evaluation_output": evaluation_output,
        "synthesis_output": synthesis_output,
        "verification_output": verification_output,
        "answer": refined_answer,
        "facts": facts
    }
    
    # Add followups if we have any
    if followups:
        result["followups"] = followups
    
    # Save the final result if session_id and session_manager provided
    if session_id and session_manager:
        session_manager.research_data["results"] = result
        session_manager.save_session()
        print(f"Saved complete results to session: {session_id}")
    
    return result

##############################################################################
# Conclusion formation and report generation
##############################################################################

def extract_thoughts_and_conclusions(text: str) -> dict:
    """
    Extracts the content inside <think>...</think> tags as 'thoughts', and the rest as 'conclusions'.
    Returns a dictionary with keys 'thoughts' and 'conclusions'.
    """
    import re
    thoughts_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    thoughts = thoughts_match.group(1).strip() if thoughts_match else ""
    conclusions = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return {"thoughts": thoughts, "conclusions": conclusions}

def generate_analysis(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Calls the LLM to generate a final analysis for the scientific question, returning a dictionary with
    'thoughts' (reasoning chain inside <think>...</think>) and 'conclusions' (summary outside those tags).
    Uses only the relevant facts for the main question.
    """
    context = filter_relevant_evidence(results['facts'], results['question'])
    analysis_prompt = (
        f"Form a comprehensive conclusion with clear reasoning chain for the scientific question:\n\n"
        f"{results['question']}\n\n"
        f"Based on all the evidence and intermediate answers we've gathered:\n\n{context}\n\n"
        f"Your conclusion should synthesize the findings, present a coherent scientific explanation, "
        f"identify key supporting evidence, acknowledge remaining uncertainties, and explain the "
        f"current scientific understanding. Focus on creating a clear chain of reasoning that "
        f"connects evidence to conclusions."
    )
    analysis_output = call_agent(
        agent_type="conclusion_formation",
        user_prompt=analysis_prompt
    )
    return extract_thoughts_and_conclusions(analysis_output)

def generate_markdown_report(export_data: Dict[str, Any]) -> str:
    """
    Generate a Markdown report from the flat export_data dict (as produced by save_research_data_to_json).
    Uses only the flat facts, main_question, conclusions, thoughts, and bibliography fields.
    """
    report = f"# Scientific Question Analysis: {export_data.get('main_question', 'Unknown Question') }\n\n"
    if 'timestamp' in export_data:
        report += f"_Generated on: {export_data['timestamp']}_\n\n"
    report += f"## Conclusion\n\n{export_data.get('conclusions', '')}\n\n"
    if export_data.get('thoughts'):
        report += f"## Reasoning Chain (Thoughts)\n\n{export_data['thoughts']}\n\n"

    # Established Facts
    report += "## Established Facts\n\n"
    facts = export_data.get('facts', [])
    if facts:
        for fact in facts:
            fact_str = f"- {fact.get('subject', '')} {fact.get('relation', '')} {fact.get('object', '')}"
            # Add evidence/source info if available
            if 'evidence' in fact and fact['evidence']:
                sources = []
                for e in fact['evidence']:
                    src = []
                    if e.get('source_id') and e.get('chunk_id'):
                        src.append(f"[{e['source_id']}{e['chunk_id']}]")
                    if e.get('source_url'):
                        src.append(f"URL: {e['source_url']}")
                    if e.get('query'):
                        src.append(f"query: '{e['query']}'")
                    if src:
                        sources.append(", ".join(src))
                if sources:
                    fact_str += f" _(evidence: {'; '.join(sources)})_"
            report += fact_str + "\n"
    else:
        report += "*No facts were extracted.*\n\n"

    # Bibliography
    report += "\n## Bibliography\n\n"
    bibliography = export_data.get('bibliography', [])
    if bibliography:
        for source in bibliography:
            report += f"- **{source.get('title', 'Untitled source')}**"
            if source.get('url'):
                report += f" ([{source['url']}]({source['url']}))"
            report += "\n"
    else:
        report += "*No sources were recorded.*\n"

    # Visited URLs (optional)
    if 'visited_urls' in export_data and export_data['visited_urls']:
        report += "\n## Visited URLs\n\n"
        for url in export_data['visited_urls']:
            report += f"- {url}\n"

    return report

def collect_bibliography_entries(result: Dict[str, Any]) -> List[Dict]:
    """Collect all bibliography entries from the result tree."""
    bibliography = []
    
    # Add sources from this level
    if "search_metadata" in result and "sources" in result["search_metadata"]:
        bibliography.extend(result["search_metadata"]["sources"])
    
    # Recursively add sources from followups
    if "followups" in result:
        for followup in result["followups"]:
            bibliography.extend(collect_bibliography_entries(followup))
    
    return bibliography

def collect_all_facts(result: Dict[str, Any], parent_name: str = None) -> List[Dict]:
    """Collect all facts from the result tree with investigation context."""
    all_facts = []
    
    # Get investigation type and name
    investigation_type = result.get("investigation_type", "unknown")
    followup_name = parent_name
    
    if "question" in result:
        if investigation_type == "followup_investigation":
            followup_name = f"Follow-up: **{result['question']}**"
    
    # Process facts at this level
    if "facts" in result:
        for fact in result["facts"]:
            fact_copy = fact.copy()
            
            # Update evidence with context if needed
            if "evidence" in fact_copy:
                for evidence in fact_copy["evidence"]:
                    if "investigation_type" not in evidence:
                        evidence["investigation_type"] = investigation_type
                    if "followup_name" not in evidence:
                        evidence["followup_name"] = followup_name
            
            all_facts.append(fact_copy)
    
    # Recursively gather facts from followups
    if "followups" in result:
        for followup in result["followups"]:
            # For sub-followups, include the parent question
            this_name = f"{followup_name} > Sub-investigation: **{followup.get('question', '')}**" if followup_name else None
            followup_facts = collect_all_facts(followup, this_name)
            
            # For each fact from the followup, either add it or merge its evidence
            for followup_fact in followup_facts:
                # Check if this fact already exists in our list
                found = False
                for existing_fact in all_facts:
                    if (existing_fact["subject"] == followup_fact["subject"] and
                        existing_fact["relation"] == followup_fact["relation"] and
                        existing_fact["object"] == followup_fact["object"]):
                        # Merge evidence
                        if "evidence" in followup_fact:
                            if "evidence" not in existing_fact:
                                existing_fact["evidence"] = []
                            existing_fact["evidence"].extend(followup_fact["evidence"])
                        found = True
                        break
                
                if not found:
                    all_facts.append(followup_fact)
    
    return all_facts


##############################################################################
# Data export function
##############################################################################

def save_research_data_to_json(question: str, results: Dict[str, Any], analysis: dict = None) -> tuple:
    """
    Save all the collected research data to a JSON file for future analysis or processing.
    If analysis is None, skip adding conclusions/thoughts.
    """
    export_data = {
        "main_question": question,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "facts": collect_all_facts(results),
        "bibliography": collect_bibliography_entries(results),
        "visited_urls": visited_urls
    }
    if analysis is not None:
        export_data["conclusions"] = analysis.get("conclusions", "")
        export_data["thoughts"] = analysis.get("thoughts", "")

    filename_base = question[:30].replace(' ', '_').replace('?', '').replace('/', '_')
    filename = f"research_data_{filename_base}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    print(f"Complete research data saved to {filename}")

    return export_data, filename

def add_analysis_to_json(filename: str, analysis: dict):
    """
    Add or update the 'conclusions' and 'thoughts' fields in the given JSON file.
    """
    import json
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data["conclusions"] = analysis.get("conclusions", "")
    data["thoughts"] = analysis.get("thoughts", "")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Analysis added to {filename}")

##############################################################################
# Main scientific question answering workflow
##############################################################################

def run_scientific_reasoning_workflow(
    scientific_question: str,
    breadth: int = 4,  # How many search queries per iteration
    depth: int = 2,    # How many layers of iteration
    agent_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    session_id: str = None,
    session_manager = None
) -> Dict[str, Any]:
    """
    Run the full scientific reasoning workflow with real-time fact verification.
    If session_id and session_manager are provided, save progress incrementally
    and resume from any existing partial results.
    """
    # Apply any agent configuration overrides
    if agent_config_overrides:
        for agent_type, override_config in agent_config_overrides.items():
            if agent_type in AGENT_CONFIGS:
                AGENT_CONFIGS[agent_type].update(override_config)
    
    # Check if we have an existing session to resume
    results = {}
    if session_id and session_manager:
        # Get research data from session
        research_data = session_manager.research_data
        
        # Check if we have partial results to resume from
        if research_data and 'results' in research_data and research_data['results']:
            print("\n=== Resuming from previous session ===")
            results = research_data['results']
            
            # If we have a complete result with followups, we're just loading
            if 'facts' in results and 'followups' in results:
                print("Session contains complete results - loading previous research.")
                
                # Save research_data for quick access
                research_data, json_filename = save_research_data_to_json(scientific_question, results, None)
                
                # Update session manager facts with flattened facts from research_data
                session_manager.facts = research_data.get('facts', [])
                session_manager.research_data['facts'] = research_data.get('facts', [])
                
                # Check if we have an analysis
                if 'conclusions' in research_data and 'thoughts' in research_data:
                    analysis = {
                        'conclusions': research_data.get('conclusions', ''),
                        'thoughts': research_data.get('thoughts', '')
                    }
                else:
                    # Generate new analysis
                    print("\n=== FORMING ANALYSIS FROM SAVED DATA ===")
                    analysis = generate_analysis({
                        'facts': research_data['facts'],
                        'question': research_data['main_question']
                    })
                    print("THOUGHTS:\n", analysis.get('thoughts', ''))
                    print("CONCLUSIONS:\n", analysis.get('conclusions', ''))
                    
                    # Add analysis to the session
                    add_analysis_to_json(json_filename, analysis)
                    
                    # Update session manager
                    session_manager.research_data['conclusions'] = analysis.get('conclusions', '')
                    session_manager.research_data['thoughts'] = analysis.get('thoughts', '')
                    session_manager.save_session()
                
                # Return the complete data
                return research_data
    
    # Print introduction
    supervisor_intro = call_agent(
        agent_type="supervisor",
        user_prompt=(
            f"We need to answer this scientific question: '{scientific_question}'. "
            f"We will use a multi-step research process involving search, information extraction, "
            f"evidence evaluation, and iterative reasoning to develop a comprehensive scientific "
            f"answer. Explain how you'll coordinate this process."
        )
    )
    print("=== SUPERVISOR INTRO ===")
    print(supervisor_intro)
    print("")

    # Add fact verification component to AGENT_CONFIGS if it doesn't exist
    if "evidence_evaluation" not in AGENT_CONFIGS:
        AGENT_CONFIGS["evidence_evaluation"] = {
            "model": DEFAULT_CONFIG["model"],
            "temperature": 0.1,  # Low temperature for factual evaluation
            "persona": "You are a scientific fact-checker. You carefully evaluate evidence to determine if facts are supported or contradicted by search results."
        }

    # Run the initial iteration with real-time fact verification
    print("\n=== Beginning research with real-time fact verification ===\n")
    
    # If we have partial results, use them; otherwise, start new
    if not results:
        results = process_iteration(
            question=scientific_question,
            context=[],
            depth_remaining=depth,
            breadth=breadth
        )
        
        # Save progress after main iteration
        if session_id and session_manager:
            session_manager.research_data['results'] = results
            session_manager.save_session()
            print(f"Progress saved to session: {session_id}")
    
    # Save complete data to JSON file (before analysis)
    research_data, json_filename = save_research_data_to_json(scientific_question, results, None)
    print(f"Complete data exported to {json_filename}")

    # Update session manager facts with flattened facts from research_data
    if session_id and session_manager:
        session_manager.facts = research_data.get('facts', [])
        session_manager.research_data['facts'] = research_data.get('facts', [])
        session_manager.save_session()
        print(f"Updated session manager with {len(session_manager.facts)} facts")

    # Form the analysis
    print("\n=== FORMING FINAL ANALYSIS ===")
    analysis = generate_analysis({
        'facts': research_data['facts'],
        'question': research_data['main_question']
    })
    print("THOUGHTS:\n", analysis.get('thoughts', ''))
    print("CONCLUSIONS:\n", analysis.get('conclusions', ''))
    print("")

    # Add analysis to the JSON file
    add_analysis_to_json(json_filename, analysis)
    
    # Update session if available
    if session_id and session_manager:
        session_manager.research_data['conclusions'] = analysis.get('conclusions', '')
        session_manager.research_data['thoughts'] = analysis.get('thoughts', '')
        
        # Automatically convert thoughts to thought nodes and create thought chain
        thoughts_text = analysis.get('thoughts', '')
        if thoughts_text:
            nodes = thoughts_text_to_nodes(thoughts_text)
            node_dicts = [node.to_dict() for node in nodes]
            if 'thought_nodes' not in session_manager.research_data:
                session_manager.research_data['thought_nodes'] = []
            session_manager.research_data['thought_nodes'].extend(node_dicts)
            
            # Create ThoughtChain and set as current
            chain = build_thought_chain(nodes)
            session_manager.research_data['current_thought_chain'] = {
                'root_id': chain.root_id
            }
            print(f"Automatically converted thoughts to {len(node_dicts)} thought nodes.")
            print(f"Created ThoughtChain with root_id: {chain.root_id}")
        
        session_manager.save_session()
        print(f"Final results saved to session: {session_id}")

    # Reload the updated JSON for the report
    with open(json_filename, 'r', encoding='utf-8') as f:
        research_data = json.load(f)

    # Generate markdown report
    print("\n=== GENERATING COMPREHENSIVE REPORT ===")
    report = generate_markdown_report(research_data)
    print("FINAL REPORT:")
    print(report)
    
    return research_data  # Return the properly structured dictionary

##############################################################################
# Entry point
##############################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_atom.py <scientific_question> [--config=path/to/config.json]")
        sys.exit(1)

    config_path = None
    agent_config_overrides = {}
    
    for arg in sys.argv[1:]:
        if arg.startswith("--config="):
            config_path = arg.split("=")[1]
            # Load configuration overrides
            try:
                with open(config_path, 'r') as f:
                    agent_config_overrides = json.load(f)
            except Exception as e:
                print(f"Error loading configuration file: {e}")
                sys.exit(1)
    
    # The scientific question is all args except the config flag
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    user_scientific_question = ' '.join(args)

    run_scientific_reasoning_workflow(
        scientific_question=user_scientific_question,
        breadth=4,
        depth=2,
        agent_config_overrides=agent_config_overrides
    )



