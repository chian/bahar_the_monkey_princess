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
    "model_id": "o3-mini",
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
        "model_id": "gpt-4.5-preview",
    },
    "query_generation": {  # Instead of "generation"
        **DEFAULT_CONFIG,
    },
    "information_extraction": {  # Instead of "reflection"
        **R1_CONFIG,
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
        "model_id": "gpt-4.5-preview",
    },
    "conclusion_formation": {  # Instead of "meta_review"
        **DEFAULT_CONFIG,
        "model_id": "gpt-4.5-preview",
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

def execute_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Use Firecrawl to get results for a scientific query.
    """
    print(f"Searching with Firecrawl: {query}")
    
    # Set 45-second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(45)
    
    try:
        # Use the simplified search method matching deep-research.ts
        response = firecrawl_app.search(
            query=query, 
            params={
                'timeout': 30000,
                'limit': num_results,
                'scrapeOptions': {'formats': ['markdown']}
            }
        )
        
        # Process results similar to deep-research.ts
        formatted_results = []
        
        # Handle different response structures
        if hasattr(response, 'data'):
            data_items = response.data
        elif isinstance(response, dict) and 'data' in response:
            data_items = response['data']
        else:
            print(f"Unexpected response format: {type(response)}")
            data_items = []
        
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
        
        # Only print response dictionary if zero results found
        if len(formatted_results) == 0:
            # Convert response to dictionary format
            if hasattr(response, '__dict__'):
                response_dict = response.__dict__
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"response": str(response)}
                
            print(f"Zero results - Response dictionary: {response_dict}")
        
        print(f"Found {len(formatted_results)} results from Firecrawl")
        
        # Track visited URLs globally
        global visited_urls
        visited_urls.extend([item["url"] for item in formatted_results if item["url"]])
        
        # Cancel the alarm
        signal.alarm(0)
        return formatted_results
        
    except TimeoutError:
        print(f"Search timed out after 45 seconds: {query}")
        return []
    except Exception as e:
        print(f"Error searching with Firecrawl: {e}")
        return []
    finally:
        # Ensure the alarm is canceled
        signal.alarm(0)

# Global list to track visited URLs for report generation
visited_urls = []

##############################################################################
# Parallel search processing, similar to deep-research.ts approach
##############################################################################

def execute_parallel_searches(queries: List[str], num_results_per_query: int = 5) -> List[Dict[str, Any]]:
    """
    Execute multiple searches in parallel with a concurrency limit
    """
    CONCURRENCY_LIMIT = 2  # Match deep-research.ts ConcurrencyLimit
    
    def search_worker(query):
        return execute_search(query, num_results_per_query)
    
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
    # Check text length - if too long, process in chunks
    MAX_SAFE_LENGTH = 120000  # Characters, not tokens, but a safe estimate
    
    if len(text) > MAX_SAFE_LENGTH:
        print(f"Text too long ({len(text)} chars), processing in chunks...")
        
        # Process text in chunks and combine results
        chunks = break_text_into_chunks(text, MAX_SAFE_LENGTH)
        all_facts = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            # Create chunk-specific source info
            chunk_source_info = source_info.copy() if source_info else {}
            if chunk_source_info:
                chunk_source_info["chunk_id"] = f"{chunk_source_info.get('chunk_id', '0')}.{i+1}"
            
            # Process this chunk
            chunk_facts = atomize_facts(chunk, chunk_source_info)
            all_facts.extend(chunk_facts)
        
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

##############################################################################
# Core iterative reasoning process
##############################################################################

def process_iteration(
    question: str, 
    context: List[str], 
    depth_remaining: int,
    breadth: int = 4,
    is_main_investigation: bool = True
) -> Dict[str, Any]:
    """
    Process one iteration of the scientific reasoning workflow.
    """
    print(f"\n=== Processing Question: {question} (Depth: {depth_remaining}) ===\n")
    
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
    
    # Format search results for the agent
    search_results_str = ""
    for i, result in enumerate(search_results):
        search_results_str += f"Result {i+1}:\n"
        search_results_str += f"Title: {result['title']}\n"
        search_results_str += f"Source: {result['source']}\n"
        search_results_str += f"URL: {result['url']}\n"
        search_results_str += f"Excerpt: {result['snippet']}\n\n"
    
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
    
    # Create full-text chunks from search results for fact extraction
    chunks_by_source = {}
    for result in search_results:
        source_id = result["source_id"]
        query = result["query"]
        
        # Get or create source in chunks_by_source
        if source_id not in chunks_by_source:
            chunks_by_source[source_id] = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "chunks": [],
                "queries": [query]
            }
        elif query not in chunks_by_source[source_id]["queries"]:
            chunks_by_source[source_id]["queries"].append(query)
        
        # Create chunk ID
        chunk_id = chr(65 + len(chunks_by_source[source_id]["chunks"]))  # A, B, C, etc.
        
        # Add content as a chunk
        chunks_by_source[source_id]["chunks"].append({
            "id": chunk_id,
            "text": result.get("content", result.get("snippet", "")),
            "query": query
        })
        
        # Add chunk to bibliography
        for source in search_metadata["sources"]:
            if source["id"] == source_id:
                source["chunks"].append({
                    "id": chunk_id,
                    "text": result.get("content", result.get("snippet", "")),
                    "query": query
                })
    
    # 7. Extract facts from the refined answer and search results
    print("\n=== Extracting Facts ===")
    
    # First extract facts from the refined answer
    answer_source_info = {
        "query": question,
        "source_id": "A",  # A for Answer
        "chunk_id": "0",
        "text": refined_answer,
        "source_url": "",  # No URL for model-generated content
        "investigation_type": "main_investigation" if is_main_investigation else "followup_investigation"
    }
    facts = atomize_facts(refined_answer, answer_source_info)
    
    # Then extract facts from each search result
    for source_id, source_data in chunks_by_source.items():
        for chunk in source_data["chunks"]:
            chunk_source_info = {
                "query": chunk["query"],
                "source_id": str(source_id),
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "source_url": source_data.get("url", ""),  # Added source URL
                "investigation_type": "main_investigation" if is_main_investigation else "followup_investigation"
            }
            
            # Extract facts from this chunk
            chunk_facts = atomize_facts(chunk["text"], chunk_source_info)
            
            # Merge with existing facts - avoid duplicates
            for new_fact in chunk_facts:
                # Check if fact already exists
                existing_fact = next((f for f in facts if 
                                    f["subject"] == new_fact["subject"] and
                                    f["relation"] == new_fact["relation"] and
                                    f["object"] == new_fact["object"]), None)
                
                if existing_fact:
                    # Merge evidence
                    if "evidence" in new_fact and new_fact["evidence"]:
                        if "evidence" not in existing_fact:
                            existing_fact["evidence"] = []
                        existing_fact["evidence"].extend(new_fact["evidence"])
                else:
                    # Add new fact
                    facts.append(new_fact)
    
    print(f"Extracted {len(facts)} facts")
    for fact in facts:
        evidence_str = ""
        if "evidence" in fact and fact["evidence"]:
            sources = [f"[{e.get('source_id', '?')}{e.get('chunk_id', '?')}]" for e in fact["evidence"]]
            evidence_str = f" (sources: {', '.join(sources)})"
        print(f"- {fact['subject']} {fact['relation']} {fact['object']}{evidence_str}")
    print("")
    
    # 8. If depth remaining, identify follow-up questions and continue
    if depth_remaining > 0:
        followup_prompt = (
            f"Based on the current answer to this question:\n\n{question}\n\n"
            f"Current answer:\n{refined_answer}\n\n"
            f"Identify {min(3, breadth)} focused follow-up questions that would help address "
            f"remaining uncertainties, explore important mechanisms, or examine alternative "
            f"explanations. Each question should contribute significant additional insight."
        )
        
        followup_output = call_agent(
            agent_type="supervisor",
            user_prompt=followup_prompt
        )
        
        parsed_questions = parse_questions(followup_output, expected_count=min(3, breadth))
        
        print("\n=== Follow-up Questions ===")
        for i, q in enumerate(parsed_questions):
            print(f"{i+1}. {q}")
        print("")
        
        # Recursive exploration of follow-up questions
        followup_results = []
        for followup in parsed_questions:
            result = process_iteration(
                followup, 
                context + [refined_answer], 
                depth_remaining - 1,
                breadth=max(2, breadth-1),  # Reduce breadth as we go deeper
                is_main_investigation=False  # Mark as not main investigation
            )
            followup_results.append(result)
        
        # Include follow-up results and search metadata in the return
        return {
            "question": question,
            "answer": refined_answer,
            "facts": facts,
            "evidence": evaluation_output,
            "followups": followup_results,
            "search_metadata": search_metadata,
            "investigation_type": "followup_investigation"
        }
    
    # Base case: return results without follow-ups
    return {
        "question": question,
        "answer": refined_answer,
        "facts": facts,
        "evidence": evaluation_output,
        "search_metadata": search_metadata,
        "investigation_type": "main_investigation" if is_main_investigation else "followup_investigation"
    }

##############################################################################
# Conclusion formation and report generation
##############################################################################

def form_conclusion(results: Dict[str, Any]) -> str:
    """
    Form a final conclusion with reasoning chain from all the iterative results.
    """
    # Format all the results into a comprehensive context
    context = format_results_to_context(results)
    
    # Generate the final conclusion with reasoning chain
    conclusion_prompt = (
        f"Form a comprehensive conclusion with clear reasoning chain for the scientific question:\n\n"
        f"{results['question']}\n\n"
        f"Based on all the evidence and intermediate answers we've gathered:\n\n{context}\n\n"
        f"Your conclusion should synthesize the findings, present a coherent scientific explanation, "
        f"identify key supporting evidence, acknowledge remaining uncertainties, and explain the "
        f"current scientific understanding. Focus on creating a clear chain of reasoning that "
        f"connects evidence to conclusions."
    )
    
    conclusion = call_agent(
        agent_type="conclusion_formation",
        user_prompt=conclusion_prompt
    )
    
    return conclusion

def generate_markdown_report(question: str, results: Dict[str, Any], conclusion: str) -> str:
    """
    Generate a comprehensive markdown report with the reasoning process.
    """
    report = f"# Scientific Question Analysis: {question}\n\n"
    report += f"## Conclusion\n\n{conclusion}\n\n"
    
    # Create a comprehensive reasoning section that builds from facts to conclusion
    report += "## Reasoning Process\n\n"
    
    # 1. Gather all atomic facts from main and follow-up investigations
    all_facts = []
    
    # Add main facts
    for fact in results['facts']:
        all_facts.append({
            'fact_text': f"{fact['subject']} {fact['relation']} {fact['object']}",
            'source': 'main investigation'
        })
    
    # Add facts from followups recursively
    def add_followup_facts(followup, source_prefix):
        for fact in followup.get('facts', []):
            all_facts.append({
                'fact_text': f"{fact['subject']} {fact['relation']} {fact['object']}",
                'source': f"{source_prefix}: {followup['question']}"
            })
        
        # Process deeper followups
        for i, sub_followup in enumerate(followup.get('followups', [])):
            add_followup_facts(sub_followup, f"{source_prefix} > Sub-investigation {i+1}")
    
    # Process all followups
    if 'followups' in results:
        for i, followup in enumerate(results['followups']):
            add_followup_facts(followup, f"Follow-up {i+1}")
    
    # 2. List all atomic facts
    report += "### Established Facts\n\n"
    for item in all_facts:
        report += f"- {item['fact_text']} _(from {item['source']})_\n"
    report += "\n"
    
    # 3. Generate reasoning chain from facts to conclusion
    report += "### Reasoning Chain\n\n"
    
    # Create a reasoning prompt for the agent to generate the chain
    reasoning_prompt = (
        f"Based on these established facts from our investigation:\n\n"
        + "\n".join([f"- {item['fact_text']}" for item in all_facts])
        + f"\n\nAnd this conclusion:\n\n{conclusion}\n\n"
        + "Create a step-by-step reasoning chain showing how these facts logically "
        + "support or lead to the conclusion. If some facts contradict parts of the conclusion, "
        + "explain the reasoning for why certain interpretations were favored over others."
    )
    
    # Generate the reasoning chain using the conclusion formation agent
    reasoning_chain = call_agent(
        agent_type="conclusion_formation",
        user_prompt=reasoning_prompt
    )
    
    report += reasoning_chain + "\n\n"
    
    # Add the original investigation details
    report += "## Investigation Details\n\n"
    
    # Add the main answer and evidence
    report += f"### Initial Investigation\n\n"
    report += f"{results['answer']}\n\n"
    report += f"**Evidence Evaluation:**\n\n{results['evidence']}\n\n"
    
    # Add follow-up investigations if they exist
    if 'followups' in results:
        report += "### Follow-up Investigations\n\n"
        for i, followup in enumerate(results['followups']):
            report += f"#### Follow-up Question {i+1}: {followup['question']}\n\n"
            report += f"{followup['answer']}\n\n"
            
            # Add nested facts
            report += "**Key Facts:**\n\n"
            for fact in followup['facts']:
                report += f"- {fact['subject']} {fact['relation']} {fact['object']}\n"
            report += "\n"
            
            # Add deeper follow-ups recursively if they exist
            if 'followups' in followup:
                report += "**Further Exploration:**\n\n"
                for j, sub_followup in enumerate(followup['followups']):
                    report += f"*Sub-question {j+1}: {sub_followup['question']}*\n\n"
                    report += f"{sub_followup['answer']}\n\n"
    
    # Add sources - using tracked URLs from searches
    report += "## Sources and References\n\n"
    global visited_urls
    unique_urls = list(set(visited_urls))
    if unique_urls:
        for url in unique_urls:
            report += f"- [{url}]({url})\n"
    else:
        report += "*No sources were tracked during this research process.*\n\n"
    
    # Add a new section for search queries
    if "search_metadata" in results and "queries" in results["search_metadata"]:
        report += "\n\n## Search Queries Used in Main Investigation\n\n"
        for i, query in enumerate(results["search_metadata"]["queries"]):
            report += f"{i+1}. `{query}`\n"
    
    # Build a complete bibliography from all investigations
    bibliography = collect_bibliography_entries(results)
    
    # Format facts with source information
    report += "\n\n## Established Facts\n\n"
    all_facts = collect_all_facts(results)
    
    for fact in all_facts:
        # Construct the fact string from components instead of expecting a 'fact' key
        fact_str = f"- {fact['subject']} {fact['relation']} {fact['object']}"
        
        # Handle multiple evidence sources
        if "evidence" in fact and fact["evidence"]:
            # Group by investigation type
            main_evidence = []
            followup_evidence = {}
            
            for evidence in fact["evidence"]:
                if evidence.get("investigation_type") == "main_investigation":
                    main_evidence.append(evidence)
                else:
                    followup_name = evidence.get("followup_name", "follow-up investigation")
                    if followup_name not in followup_evidence:
                        followup_evidence[followup_name] = []
                    followup_evidence[followup_name].append(evidence)
            
            # Start with sources info
            sources_info = []
            
            # Format main investigation sources with URLs
            if main_evidence:
                main_citations = [f"[{e['source_id']}{e['chunk_id']}]" for e in main_evidence]
                main_queries = list(set([e['query'] for e in main_evidence]))
                main_urls = list(set([e['source_url'] for e in main_evidence if e.get('source_url')]))
                
                main_info = "from main investigation"
                if main_queries:
                    query_list = ", ".join([f'"{q}"' for q in main_queries])
                    main_info += f", queries: {query_list}"
                
                if main_citations:
                    main_info += f", sources: {', '.join(main_citations)}"
                
                if main_urls:
                    main_info += f", URLs: {', '.join(main_urls)}"
                
                sources_info.append(main_info)
            
            # Format followup investigation sources
            for followup_name, evidence_list in followup_evidence.items():
                followup_citations = [f"[{e['source_id']}{e['chunk_id']}]" for e in evidence_list]
                followup_queries = list(set([e['query'] for e in evidence_list]))
                
                followup_info = f"from {followup_name}"
                if followup_queries:
                    query_list = ", ".join([f'"{q}"' for q in followup_queries])
                    followup_info += f", queries: {query_list}"
                
                if followup_citations:
                    followup_info += f", sources: {', '.join(followup_citations)}"
                
                sources_info.append(followup_info)
            
            # Format all sources info
            if sources_info:
                fact_str += f" _({'; '.join(sources_info)})_"
        
        report += fact_str + "\n"
    
    # Add bibliography section
    report += "\n\n## Bibliography\n\n"
    for source in bibliography:
        report += f"{source['id']}. **{source['title']}**\n"
        if "url" in source and source["url"]:
            report += f"   URL: {source['url']}\n"
        
        # Add text chunks with their IDs
        if "chunks" in source:
            for chunk in source["chunks"]:
                report += f"\n   [{chunk['id']}] {chunk['text'][:200]}..."
                if len(chunk['text']) > 200:
                    report += " _(truncated)_"
                report += "\n"
        
        report += "\n"
    
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

def save_research_data_to_json(question: str, results: Dict[str, Any], conclusion: str) -> str:
    """
    Save all the collected research data to a JSON file for future analysis or processing.
    Includes questions, answers, facts, evidence, bibliography, and all search queries.
    
    Args:
        question: The main scientific question
        results: The full results dictionary from process_iteration
        conclusion: The final conclusion text
        
    Returns:
        The filename of the saved JSON file
    """
    # Create a complete data export including everything
    export_data = {
        "main_question": question,
        "conclusion": conclusion,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "facts": collect_all_facts(results),
        "bibliography": collect_bibliography_entries(results),
        "visited_urls": visited_urls
    }
    
    # Create a clean filename based on the question
    filename_base = question[:30].replace(' ', '_').replace('?', '').replace('/', '_')
    filename = f"research_data_{filename_base}.json"
    
    # Save the JSON file with indentation for readability
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Complete research data saved to {filename}")
    return filename

##############################################################################
# Main scientific question answering workflow
##############################################################################

def run_scientific_reasoning_workflow(
    scientific_question: str,
    breadth: int = 4,  # How many search queries per iteration
    depth: int = 2,    # How many layers of iteration
    agent_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None
) -> str:
    """
    Instead of generating hypotheses, this function orchestrates a multi-round 
    scientific reasoning workflow to answer a specific scientific question.
    """
    # Apply any agent configuration overrides
    if agent_config_overrides:
        for agent_type, override_config in agent_config_overrides.items():
            if agent_type in AGENT_CONFIGS:
                AGENT_CONFIGS[agent_type].update(override_config)
    
    # Supervisor introduces the process
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

    # Run the initial iteration
    results = process_iteration(
        question=scientific_question,
        context=[],
        depth_remaining=depth,
        breadth=breadth
    )
    
    # Form the conclusion
    print("\n=== FORMING FINAL CONCLUSION ===")
    conclusion = form_conclusion(results)
    print(conclusion)
    print("")
    
    # Generate markdown report
    print("\n=== GENERATING COMPREHENSIVE REPORT ===")
    report = generate_markdown_report(scientific_question, results, conclusion)
    print("Report generated successfully!")
    
    # Save report to file
    report_filename = f"scientific_report_{scientific_question[:30].replace(' ', '_')}.md".replace('?', '')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {report_filename}")
    
    # Save complete data to JSON file
    json_filename = save_research_data_to_json(scientific_question, results, conclusion)
    print(f"Complete data exported to {json_filename}")
    
    return conclusion

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


