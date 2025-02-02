from langchain_openai import AzureChatOpenAI
import os 
from dotenv import load_dotenv, find_dotenv
from utils_discussion import AgentState, SummarisationRequest, SummarisationResponse, MemoryManagementRequest, MemoryManagementResponse, QuestionGenerationRequest, QuestionGenerationResponse
from datetime import datetime
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

import logging

# Summarisation Agent
def create_summarisation_agent():
    def summarize(state: AgentState) -> AgentState:
        # Get latest discussion
        # latest_discussion = state.discussions
        llm_summary = AzureChatOpenAI(
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
            model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
            api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            temperature=0,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
        )
        logging.info(f"LLM model initialized")

        messages = [
            SystemMessage(content="""You are a discussion summarizer. 
            Create a concise summary of the discussion, extracting key points 
            and insights from the answers. Focus on patterns, unique perspectives, 
            and potential areas for deeper exploration."""),
            
            HumanMessage(content=f"""
            Question discussed: {state.current_question}
            
            Answers from participants:
            {state.discussions}
            
            Provide a short summary of the discussion <max 50 words>.
            """)
        ]

        llm_structured_output_summary = llm_summary.with_structured_output(SummarisationResponse)
        response = llm_structured_output_summary.invoke(messages)
        logging.info(f"response create_summarisation_agent: {response}")
        print("response create_summarisation_agent: ", response)
        timestamp = str(datetime.now().isoformat())
        state.summaries.append({timestamp: response.summary})
        return state
    return summarize
    

def create_memory_management_agent():
    def memory_management(state: AgentState) -> AgentState:
        llm_memory = AzureChatOpenAI(
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
            model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
            api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            temperature=0,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
        )
        # logging.info(f"LLM model initialized")
        latest_summary = state.summaries[-1]
        messages = [
            SystemMessage(content="""You are a memory manager for an ongoing discussion.
            Analyze the new summary and existing memories to:
            1. Identify important points to remember
            2. Determine which old memories are still relevant
            3. Suggest connections between past and present discussions
            4. Consider the temporal relevance of memories"""),
            
            HumanMessage(content=f"""
            New Summary: {latest_summary}
            
            Existing Memories:
            {state.memories}
            
            Determine which memories to keep, update, or create.

            Return the updated memories
            """)
        ]
        llm_structured_output_memory = llm_memory.with_structured_output(MemoryManagementResponse)
        response = llm_structured_output_memory.invoke(messages)

        # print("response create_memory_management_agent: ", response)
        # logging.info(f"response create_memory_management_agent: {response}")
        timestamp = str(datetime.now().isoformat())
        if len(state.memories) > 0:
            state.memories.pop(0)
        state.memories.append({timestamp:response.new_memories})
        return state
    return memory_management
    
def create_question_generation_agent():
    def generate_question(state: AgentState) -> AgentState:
        llm_question = AzureChatOpenAI(
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
            model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
            api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            temperature=0,
            max_tokens=100,
            timeout=None,
            max_retries=2,
        )
        # logging.info(f"LLM model initialized")

        recent_summaries = state.summaries[-2:]
        current_memories = state.memories 
        previous_question = state.current_question
        messages = [
            SystemMessage(content="""You are a discussion facilitator generating 
            thought-provoking follow-up questions. Consider recent discussions,
            key memories, and the previous question to create an engaging
            question that deepens the conversation."""),
            
            HumanMessage(content=f"""
            Previous Question: {previous_question}
            
            Recent Discussion Summaries:
            {recent_summaries}
            
            Relevant Memories:
            {current_memories}
            
            Generate a thought-provoking follow-up question in less than 100 words. 
            """)
        ]
        llm_structured_output_question = llm_question.with_structured_output(QuestionGenerationResponse)
        response = llm_structured_output_question.invoke(messages)
        # logging.info(f"response create_question_generation_agent: {response}")
        # print("response create_question_generation_agent: ", response)
        state.next_question = response.next_question
        return state
    return generate_question

def create_discussion_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("summarize", create_summarisation_agent())
    workflow.add_node("manage_memory", create_memory_management_agent())
    workflow.add_node("generate_question", create_question_generation_agent())
    
    #define edges
    workflow.add_edge("summarize", "manage_memory")
    workflow.add_edge("manage_memory", "generate_question")
    
    # Set entry point
    workflow.set_entry_point("summarize")
    
    # Set exit point
    workflow.set_finish_point("generate_question")
    
    return workflow.compile()

