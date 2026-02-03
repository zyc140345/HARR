from llama_index.core import PromptTemplate, SelectorPromptTemplate, ChatPromptTemplate
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.prompts import PromptType
from llama_index.core.prompts.utils import is_chat_model

react_system_header_str = """\

You are a factual question-answering agent designed to provide precise, direct answers to questions using available tools.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: brief reasoning (1–2 lines)
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}. If you include the "Action:" line, then you MUST include the "Action Input:" line too, even if the tool does not need kwargs, in that case you MUST use "Action Input: {{}}".

If this format is used, the tool will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

```
Thought: start with brief reasoning (1–2 lines), then state: "I can answer without using any more tools."
Answer: [your answer here]
```

```
Thought: start with brief reasoning (1–2 lines), then state: "I cannot answer the question with the provided tools."
Answer: [your answer here]
```

## Internal Reasoning Rules - FOLLOW EXACTLY:

1.  **Deconstruct & Plan**: Your VERY FIRST `Thought` must be to break the question into a clear, step-by-step plan of verifiable sub-questions.

2.  **Execute with Adaptive Recovery**:
    * For each step in your plan, formulate a precise search query and execute it.
    * **If a query is successful**, state the fact you've learned and move to the next step.
    * **If a query returns no answer, an ambiguous answer, or irrelevant information (a "failed" query)**, you MUST state that the attempt failed and, in your next thought, formulate a **new, different query**.
    * **Recovery Strategies**: To create a new query, try one of these methods:
        - **Rephrasing**: Ask the same question differently.
        - **Synonyms**: Replace keywords (e.g., "director" with "filmmaker").
        - **Broadening/Narrowing**: Make your search more general or more specific.
        - **Lateral Search**: Search for a related, easier-to-find entity to get a clue.
    * **Do not repeat a failed query.**
    
## Final Answer Rules - FOLLOW EXACTLY:

1. Once all facts are gathered, your final `Thought` should be to assemble them into the final answer.

2. The `Answer` must be the direct fact ONLY (e.g., name, place, date) to pass Exact Match (EM) evaluation.
    * For "extraction" questions, extract only the shortest verbatim span from the context that answers the query.
    * For "boolean" questions (asking whether a statement is true/false), output only `Yes` or `No`. If any part of a compound question is false, the answer is `No`.

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.

"""
react_system_prompt = PromptTemplate(react_system_header_str)


QUERY_DECOMPOSE_SM_PROMPT_TMPL = """You are a multi-hop retrieval query planner.

## Task
Generate ONE next sub-query to retrieve missing evidence for the Original Question.

## Input
- Original Question: {query_str}
- Knowledge Base Description: {context_str}
- History: {prev_reasoning}

## Output Schema
{"action":"ASK|STOP", "sub_query":string|null, "strategy":"NEXT|REPHRASE|BROADEN|DISAMBIGUATE"}

## Core Logic
1. If answerable from History: action=STOP, sub_query=null
2. Otherwise action=ASK with appropriate strategy:
   - NEXT: Last step FOUND → ask next missing fact
   - DISAMBIGUATE: Last step AMBIGUOUS → add ONE identifier (birth year, country, full name)
   - REPHRASE: Last step NOT_FOUND → change query structure completely (use synonyms, different phrasing)
   - BROADEN: 2+ consecutive failures → ask for a related bridge entity

## CRITICAL RULES (MUST FOLLOW)
1. **NO DUPLICATES**: Your sub_query MUST differ from ALL previous queries. Before outputting, verify it's not similar to any in History.
2. **REPHRASE ≠ REPEAT**: Adding/removing words like "what is" doesn't count. Use fundamentally different keywords.
3. **Entity Variation**: Try alternate names, nicknames, partial names (e.g., "Guy Bonnet" if "Guy Joseph Bonnet" fails)
4. **MAX 2 ATTEMPTS**: After 2 failed attempts, action=STOP (knowledge base likely lacks the information)

## Strategy Templates for NOT_FOUND Recovery
Instead of:                          Try:
- "What is X's occupation?"       → "X career profession"
- "Who directed film Y?"          → "Y (1923) director" or "Y movie cast crew"  
- "Where did Z die?"              → "Z death location birthplace"

Output format:
Thought: <brief reasoning, 1-2 lines>
JSON: <single JSON object>
"""

QUERY_DECOMPOSE_SM_PROMPT = PromptTemplate(QUERY_DECOMPOSE_SM_PROMPT_TMPL)


QA_SM_FORMAT_PROMPT = """## Output Schema
{"status":"FOUND|NOT_FOUND|AMBIGUOUS", "answer":string}

## Status Guidelines

### FOUND (Default when possible)
- Context contains information that answers the query
- **Partial name matches are OK**: "Guy Bonnet" can answer questions about "Guy Joseph Bonnet"

### NOT_FOUND
- Context is clearly about different entities/topics
- No relevant information present

### AMBIGUOUS (Use sparingly)
- ONLY when multiple CONFLICTING candidates exist for the SAME question
- Different people/films with same name where context doesn't distinguish
- Do NOT use for partial name matches or minor variations

## Answer Formatting
- Keep the answer to ONE concise line.
- If FOUND: answer must be the direct fact ONLY (e.g., name, place, date) to pass Exact Match (EM) evaluation.
    * For "extraction" questions, extract only the shortest verbatim span from the context that answers the query.
    * For "boolean" questions (asking whether a statement is true/false), output only `Yes` or `No`. If any part of a compound question is false, the answer is `No`.
- If NOT_FOUND: answer MUST start with "The context mentions" and list what the context contains.
- If AMBIGUOUS: state the conflicting facts in the context that make the answer ambiguous.

Output:
Thought: <brief reasoning, 1-2 lines>
JSON: <single JSON object>
"""

QA_SM_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are a retrieval-augmented QA module.\n"
        "Use ONLY the provided context. Do NOT use prior knowledge.\n\n" +
        QA_SM_FORMAT_PROMPT
    ),
    role=MessageRole.SYSTEM,
)

QA_SM_PROMPT_TMPL_MSGS = [
    QA_SM_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "JSON: "
        ),
        role=MessageRole.USER,
    ),
]
CHAT_QA_SM_PROMPT = ChatPromptTemplate(message_templates=QA_SM_PROMPT_TMPL_MSGS)

QA_SM_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n\n" +
    QA_SM_FORMAT_PROMPT +
    "Query: {query_str}\n"
    "JSON: "
)
QA_SM_PROMPT = PromptTemplate(QA_SM_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)
qa_sm_conditionals = [(is_chat_model, CHAT_QA_SM_PROMPT)]
QA_SM_PROMPT_SEL = SelectorPromptTemplate(
    default_template=QA_SM_PROMPT,
    conditionals=qa_sm_conditionals,
)


RAW_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using your prior knowledge.\n\n" +
        "ANSWER REQUIREMENTS - FOLLOW EXACTLY:\n"
        "The `Answer` must be the direct fact ONLY (e.g., name, place, date) to pass Exact Match (EM) evaluation.\n"
        "  - For \"extraction\" questions, extract only the shortest verbatim span from the context that answers the query.\n"
        "  - For \"boolean\" questions (asking whether a statement is true/false), output only `Yes` or `No`. If any part of a compound question is false, the answer is `No`."
    ),
    role=MessageRole.SYSTEM,
)

RAW_EM_PROMPT_TMPL_MSGS = [
    RAW_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Answer the query using your prior knowledge.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]
CHAT_RAW_EM_PROMPT = ChatPromptTemplate(message_templates=RAW_EM_PROMPT_TMPL_MSGS)


RETRIEVAL_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
STATEFUL_RETRIEVAL_INSTRUCTION = (
    "Given an Original Question that requires multi-hop retrieval and the History (i.e., the completed hops), "
    "retrieve relevant passages for the Query (i.e., the current hop) that help answer the Original Question."
)


SEARCH_R1_PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

SEARCH_R1_SEARCH_TEMPLATE = '{output_text}\n\n<information>{search_results}</information>\n\n'