import sys
import openai
from typing import List, Dict, Any, Optional, Union, Callable
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

##############################################################################
# Configuration: Define default and agent-specific model information
##############################################################################

# Default configuration
DEFAULT_CONFIG = {
    "api_key": os.getenv('OPENAI_API_KEY'),
    "endpoint": "https://api.openai.com/v1",
    "model_id": "o3-mini",
    "max_completion_tokens": 16000,
    "tools": []
}
R1_CONFIG = {
    "api_key": os.getenv('TOGETHER_API_KEY'),
    "endpoint": "https://api.together.xyz/v1",
    "model_id": "deepseek-ai/DeepSeek-R1",
    "max_completion_tokens": 16000,
    "tools": []
}

# Agent-specific configurations - override as needed
AGENT_CONFIGS = {
    "supervisor": {
        **DEFAULT_CONFIG,
        "model_id": "gpt-4.5-preview",  # Higher quality model for coordination
    },
    "generation": {
        **DEFAULT_CONFIG,
    },
    "reflection": {
        **R1_CONFIG,
    },
    "ranking": {
        **DEFAULT_CONFIG,
    },
    "evolution": {
        **DEFAULT_CONFIG,
    },
    "proximity_check": {
        **DEFAULT_CONFIG,
    },
    "meta_review": {
        **DEFAULT_CONFIG,
        "model_id": "gpt-4.5-preview",  # Higher quality model for final review
        "max_completion_tokens": 2000,  # More tokens for comprehensive review
    },
    # Example of a different model/endpoint configuration:
    # "alternative_agent": {
    #     "api_key": "CELS",
    #     "endpoint": "http://195.88.24.64:80/v1",
    #     "model_id": "meta-llama/Llama-3.3-70B-Instruct",
    #     "max_tokens": 1500,
    #     "tools": []
    # }
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
            "You are the Supervisor Agent in a multi-agent AI co-scientist system. "
            "You receive the high-level research goal and coordinate a series of "
            "agents (Generation, Reflection, Ranking, Evolution, Proximity Check, "
            "Meta-review) to iteratively produce, refine, and rank scientific ideas. "
            "Manage the overall workflow and produce final instructions or summaries "
            "to the user. Keep track of each round, pass the correct context to each "
            "agent, and store or update the system's memory as needed."
        )
    elif agent_type == "generation":
        prompt = (
            "You are the Generation Agent in a multi-agent AI co-scientist system. "
            "You produce new ideas and hypotheses in response to a defined "
            "research goal. For each idea, you MUST include an explicit hypothesis "
            "statement. Leverage existing literature, domain knowledge, and "
            "creative thinking to propose multiple distinct research directions, "
            "frameworks, or experimental designs. Strive for novelty, practicality, "
            "and scientific rigor."
        )
    elif agent_type == "reflection":
        prompt = (
            "You are the Reflection Agent in a multi-agent AI co-scientist system. "
            "You critically analyze each proposed idea and its hypothesis. For each, "
            "evaluate plausibility, novelty, potential flaws, and likelihood of being correct. "
            "Recommend improvements or missing angles, and highlight strengths and weaknesses "
            "so that subsequent agents can refine the ideas further."
        )
    elif agent_type == "ranking":
        prompt = (
            "You are the Ranking Agent in a multi-agent AI co-scientist system. "
            "You receive multiple research ideas or proposals, each containing an explicit "
            "hypothesis. Your job is to compare them, simulate a debate about their merits, "
            "and rank them from most to least promising. In the rationale, highlight: "
            "(1) Hypothesis plausibility, (2) Novelty, and (3) Likelihood of correctness. "
            "Use these criteria to produce a final ranking."
        )
    elif agent_type == "evolution":
        prompt = (
            "You are the Evolution Agent in a multi-agent AI co-scientist system. "
            "You take an existing set of research ideas (each with a hypothesis) and "
            "refine or evolve them. You may simplify complex designs, combine ideas, "
            "or extend them into new directions, but each refined idea must retain an "
            "explicit hypothesis. Highlight key changes so the ideas become stronger, "
            "more innovative, or more feasible."
        )
    elif agent_type == "proximity_check":
        prompt = (
            "You are the Proximity Check Agent in a multi-agent AI co-scientist system. "
            "Your role is to evaluate whether newly generated or revised ideas stay "
            "aligned with the assigned research goal, meet ethical and feasibility "
            "constraints, and do not drift too far from the desired objectives. If "
            "misalignment is detected, provide warnings and corrective suggestions."
        )
    elif agent_type == "meta_review":
        prompt = (
            "You are the Meta-review Agent in a multi-agent AI co-scientist system. "
            "You take the final set of refined, top-ranked research proposals (only "
            "the top 5 in the final ranking) and compose a meta-analysis: summarize "
            "the core ideas, discuss strengths and limitations, and suggest practical "
            "next steps. Provide a concise but comprehensive overview."
        )
    else:
        prompt = (
            f"You are the {agent_type} Agent in a multi-agent AI co-scientist system. "
            "Provide expert assistance based on your specialized role."
        )
    
    return config, prompt

##############################################################################
# Tool execution framework
##############################################################################

def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """
    Executes a tool based on the tool name and input parameters.
    This is a placeholder for a more sophisticated tool execution system.
    """
    if tool_name == "literature_search":
        return f"Literature search results for: {tool_input.get('query', '')}"
    elif tool_name == "data_analysis":
        return f"Data analysis results for: {tool_input.get('dataset', '')}"
    elif tool_name == "citation_check":
        return f"Citation information for: {tool_input.get('paper', '')}"
    else:
        return f"Tool {tool_name} not implemented or not found."

##############################################################################
# Helper function to call an agent via the OpenAI ChatCompletion API
##############################################################################

def call_agent(
    agent_type: str,
    user_prompt: str,
    additional_context: str = "",
    override_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call an agent using its configuration and prompt.
    
    Parameters:
    - agent_type: The type of agent to call (e.g., "supervisor", "generation")
    - user_prompt: The user's instruction or query
    - additional_context: Optional context from previous exchanges
    - override_config: Optional configuration overrides for this specific call
    
    Returns:
    - The agent's response text
    """
    # Get agent configuration and prompt
    config, system_prompt = get_agent_config_and_prompt(agent_type)
    
    # Apply any configuration overrides
    if override_config:
        for key, value in override_config.items():
            config[key] = value
    
    # Prepare the messages
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if additional_context:
        messages.append({"role": "assistant", "content": additional_context})

    messages.append({"role": "user", "content": user_prompt})
    
    # Prepare API parameters
    api_params = {
        "model": config["model_id"],
        "messages": messages,
    }
    
    # Add the token limit parameter as specified in the config
    if "max_completion_tokens" in config:
        api_params["max_completion_tokens"] = config["max_completion_tokens"]
    elif "max_tokens" in config:
        api_params["max_tokens"] = config["max_tokens"]
    
    # Add tools if specified in the configuration
    tools = config.get("tools", [])
    if tools:
        api_params["tools"] = tools
    
    # Create client using the agent-specific configuration
    client = openai.OpenAI(
        api_key=config["api_key"], 
        base_url=config["endpoint"]
    )
    
    # Make the API call
    response = client.chat.completions.create(**api_params)
    
    # Handle tool calls if present
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
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
        
        # Make a second API call with the tool results
        follow_up_response = client.chat.completions.create(**api_params)
        return follow_up_response.choices[0].message.content
    
    return response.choices[0].message.content

##############################################################################
# Main multi-round workflow function
##############################################################################

def run_co_scientist_workflow(
    research_goal: str,
    num_ideas: int = 10,
    num_rounds: int = 3,
    agent_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """
    This function orchestrates a multi-round AI co-scientist workflow with modular agent configurations.
    
    Parameters:
    - research_goal: The research objective
    - num_ideas: Number of ideas to generate per round
    - num_rounds: Number of iterative rounds
    - agent_config_overrides: Optional dictionary to override configurations for specific agents
    """
    # Apply any agent configuration overrides
    if agent_config_overrides:
        for agent_type, override_config in agent_config_overrides.items():
            if agent_type in AGENT_CONFIGS:
                AGENT_CONFIGS[agent_type].update(override_config)
    
    # Supervisor welcomes and sets the stage
    supervisor_intro = call_agent(
        agent_type="supervisor",
        user_prompt=(
            f"The user has the research goal: '{research_goal}'. "
            "We will conduct multiple rounds of idea generation and refinement, "
            "ensuring each idea has an explicit hypothesis. We will remove weaker "
            "proposals each round and replace them. Eliminated proposals must not "
            "reappear in subsequent rounds."
        )
    )
    print("=== SUPERVISOR INTRO ===")
    print(supervisor_intro)
    print("")

    # Keep track of the ideas across rounds
    ideas: List[str] = []
    excluded_ideas: List[str] = []  # Track eliminated ideas so they do not reappear

    # Define the fraction of ideas to remove each round
    removal_fractions = [0.5, 1/3, 0.25]

    for round_idx in range(num_rounds):
        print(f"\n========== ROUND {round_idx+1} / {num_rounds} ==========\n")

        # 1) Generate or Evolve Ideas
        if round_idx == 0:
            # First round, generate new ideas
            gen_prompt = (
                f"Please generate {num_ideas} distinct research ideas or hypotheses "
                f"for the goal: '{research_goal}'. For each idea, include an explicit "
                f"hypothesis. Avoid any ideas that are in this excluded list: {excluded_ideas}."
            )
            generation_output = call_agent(
                agent_type="generation",
                user_prompt=gen_prompt
            )
            ideas = parse_ideas_from_text(generation_output, expected_count=num_ideas)
            print("=== GENERATION AGENT OUTPUT ===")
            print(generation_output)
            print("")
        else:
            # In subsequent rounds, we first remove the weakest proposals, then refine
            # existing ones, then generate new ones to replace the removed ones.

            # Evolve existing ideas
            print("=== EVOLUTION AGENT OUTPUT (Refining Existing Ideas) ===")
            ideas_text = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
            evolve_prompt = (
                f"We have the following {len(ideas)} ideas:\n\n"
                f"{ideas_text}\n\n"
                "Please refine or evolve each idea to be stronger, more novel, or more feasible. "
                "Each must retain an explicit hypothesis. Avoid introducing any idea that's in "
                f"this excluded list: {excluded_ideas}."
            )
            evolution_output = call_agent(
                agent_type="evolution",
                user_prompt=evolve_prompt
            )
            evolved_ideas = parse_ideas_from_text(
                evolution_output,
                expected_count=len(ideas)
            )
            print(evolution_output)
            print("")

            ideas = evolved_ideas

        # 2) Reflect on the ideas
        ideas_text = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
        reflection_prompt = (
            f"Please analyze these {len(ideas)} ideas, each with its hypothesis, "
            "for plausibility, novelty, potential flaws, and likelihood of being correct:\n\n"
            + ideas_text
        )
        reflection_output = call_agent(
            agent_type="reflection",
            user_prompt=reflection_prompt
        )
        print("=== REFLECTION AGENT OUTPUT ===")
        print(reflection_output)
        print("")

        # 3) Proximity Check
        proximity_prompt = (
            f"Please ensure these ideas remain aligned with the research goal '{research_goal}' "
            "and check for ethical, feasibility, or scope concerns. If any are out of scope, "
            "suggest modifications or indicate if they should be dropped:\n\n" + ideas_text
        )
        proximity_output = call_agent(
            agent_type="proximity_check",
            user_prompt=proximity_prompt
        )
        print("=== PROXIMITY CHECK AGENT OUTPUT ===")
        print(proximity_output)
        print("")

        # 4) Ranking
        ranking_prompt = (
            f"We have these {len(ideas)} ideas:\n\n{ideas_text}\n\n"
            "Please rank them from most promising to least promising, "
            "considering (1) Hypothesis plausibility, (2) Novelty, and "
            "(3) Likelihood of correctness. Provide a rationale."
        )
        ranking_output = call_agent(
            agent_type="ranking",
            user_prompt=ranking_prompt
        )
        print("=== RANKING AGENT OUTPUT ===")
        print(ranking_output)
        print("")

        # Reorder `ideas` by the ranking
        ideas_ordered = parse_ideas_order_from_ranking(ranking_output, ideas)
        ideas = ideas_ordered

        # Remove the weaker proposals based on the fraction for the current round
        fraction = removal_fractions[round_idx] if round_idx < len(removal_fractions) else 0.25
        num_to_remove = int(len(ideas) * fraction)
        if num_to_remove < 1 and len(ideas) > 1:
            num_to_remove = 1

        if num_to_remove > 0:
            removed_ideas = ideas[-num_to_remove:]  # take from the bottom
            ideas = ideas[:-num_to_remove]
            excluded_ideas.extend(removed_ideas)

            # Generate new ideas to replace them
            gen_prompt = (
                f"Please generate {num_to_remove} new distinct research ideas. Each must include "
                "an explicit hypothesis, and they must not reintroduce or duplicate any in this "
                f"excluded list: {excluded_ideas}. The research goal is: '{research_goal}'."
            )
            new_generation_output = call_agent(
                agent_type="generation",
                user_prompt=gen_prompt
            )
            new_ideas = parse_ideas_from_text(new_generation_output, expected_count=num_to_remove)
            ideas.extend(new_ideas)

            print("=== REMOVING WEAKER PROPOSALS ===")
            print(f"We removed {num_to_remove} ideas (the weakest):")
            for idx, removed in enumerate(removed_ideas):
                print(f"- {removed}")
            print("\n=== GENERATION AGENT OUTPUT (Replacement Ideas) ===")
            print(new_generation_output)
            print("")

        # Supervisor summary of the round
        round_summary = call_agent(
            agent_type="supervisor",
            user_prompt=(
                f"Summarize the results of round {round_idx+1}, referencing the Reflection, "
                f"Proximity Check, and Ranking. The final set of ideas (after removal/replacement) "
                f"are:\n\n"
                + "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
            )
        )
        print("=== SUPERVISOR ROUND SUMMARY ===")
        print(round_summary)

    # After all rounds, do a final ranking and then a Meta-review of only the top 5
    ideas_text = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
    final_ranking_prompt = (
        f"Before the final Meta-review, please rank these {len(ideas)} ideas again, "
        "from most promising to least promising, to ensure we present the top 5. "
        "Focus on (1) Hypothesis plausibility, (2) Novelty, and (3) Likelihood of correctness.\n\n"
        + ideas_text
    )
    final_ranking_output = call_agent(
        agent_type="ranking",
        user_prompt=final_ranking_prompt
    )
    print("\n=== FINAL RANKING AGENT OUTPUT ===")
    print(final_ranking_output)

    # Reorder `ideas` by the final ranking
    final_ideas_ordered = parse_ideas_order_from_ranking(final_ranking_output, ideas)
    ideas = final_ideas_ordered

    # Keep only top 5
    top_5_ideas = ideas[:5]

    # Meta-review on only the top 5
    final_ideas_text = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(top_5_ideas)])
    meta_prompt = (
        f"Here are the final top {len(top_5_ideas)} ideas from the iterative process:\n\n"
        f"{final_ideas_text}\n\n"
        "Please provide a meta-review, summarizing the best ideas, their strengths "
        "and weaknesses, and suggest next steps for the scientist."
    )
    meta_review_output = call_agent(
        agent_type="meta_review",
        user_prompt=meta_prompt
    )
    print("\n=== META-REVIEW AGENT OUTPUT (TOP 5 ONLY) ===")
    print(meta_review_output)

##############################################################################
# Simple Parsers for Idea Lists and Rankings
##############################################################################

def parse_ideas_from_text(text: str, expected_count: int) -> List[str]:
    """
    Very naive parser to try to split the text into 'expected_count' ideas.
    In a real system, you'd want a more robust approach.
    """
    lines = text.split("\n")
    lines = [ln.strip() for ln in lines if ln.strip()]

    ideas = []
    current_idea = []
    for line in lines:
        if is_new_idea_start(line):
            # If we have a current_idea in progress, push it
            if current_idea:
                ideas.append(" ".join(current_idea).strip())
                current_idea = []
            current_idea.append(line)
        else:
            current_idea.append(line)

    if current_idea:
        ideas.append(" ".join(current_idea).strip())

    # Clamp to expected_count if needed
    if len(ideas) > expected_count:
        ideas = ideas[:expected_count]

    cleaned_ideas = [remove_leading_number(idea).strip() for idea in ideas]

    return cleaned_ideas

def parse_ideas_order_from_ranking(ranking_output: str, current_ideas: List[str]) -> List[str]:
    """
    A naive approach to reorder `current_ideas` based on the Ranking Agent output.
    Attempt to detect numeric order from the ranking output text.
    """
    new_order = []
    lines = ranking_output.split("\n")
    idea_map = {i+1: current_ideas[i] for i in range(len(current_ideas))}

    for line in lines:
        line_stripped = line.strip()
        if line_stripped and line_stripped[0].isdigit():
            # e.g. "1. Idea about X"
            idx_str = line_stripped.split(".")[0].strip()
            try:
                idx = int(idx_str)
                if idx in idea_map and idea_map[idx] not in new_order:
                    new_order.append(idea_map[idx])
            except ValueError:
                pass

    # Append any missing ideas in the original order
    for i in range(len(current_ideas)):
        if idea_map[i+1] not in new_order:
            new_order.append(idea_map[i+1])

    return new_order

def is_new_idea_start(line: str) -> bool:
    line_stripped = line.strip()
    if len(line_stripped) < 2:
        return False
    if line_stripped[0].isdigit() and line_stripped[1] in [".", ")"]:
        return True
    if line_stripped.startswith("- "):
        return True
    return False

def remove_leading_number(text: str) -> str:
    import re
    return re.sub(r'^\s*\d+[\.\)]\s*', '', text).strip()

##############################################################################
# Run the workflow from a command-line argument
##############################################################################

if __name__ == "__main__":
    import json  # Add import for JSON handling needed for tool calls
    
    if len(sys.argv) < 2:
        print("Usage: python co_scientist.py <research_goal> [--config=path/to/config.json]")
        sys.exit(1)

    # Check if there's a config flag
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
    
    # The research goal is all args except config ones
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    user_research_goal = ' '.join(args)
    
    # Example of setting different configurations for different agents:
    # agent_config_overrides = {
    #     "generation": {
    #         "api_key": "ALTERNATIVE_KEY",
    #         "endpoint": "http://alternative-api.example.com/v1",
    #         "model_id": "alternative-model",
    #         "temperature": 1.0  # Higher temperature for more creative generation
    #     }
    # }
    
    run_co_scientist_workflow(
        research_goal=user_research_goal,
        num_ideas=10,   # Default number of ideas per round
        num_rounds=3,    # Default number of iterative rounds
        agent_config_overrides=agent_config_overrides
    )
