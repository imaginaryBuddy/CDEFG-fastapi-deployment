from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# 1. First, define our State
class TripPlannerState(BaseModel):
    """State management for Trip Planning Assistant"""
    # Core user information
    current_conversation: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Current conversation history"
    )
    
    # Extracted preferences
    preferences: Dict[str, any] = Field(
        default_factory=dict,
        description="Extracted travel preferences"
    )
    
    # Trip details
    trip_details: Dict[str, any] = Field(
        default_factory=dict,
        description="Specific trip details like dates, budget, etc."
    )
    
    # Generated recommendations
    recommendations: List[Dict[str, any]] = Field(
        default_factory=list,
        description="Generated travel recommendations"
    )
    
    # Memory of previous interactions
    memories: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Important points to remember about user preferences"
    )

# 2. Create our agents
def create_preference_extraction_agent():
    """Agent that extracts travel preferences from conversation"""
    def extract_preferences(state: TripPlannerState) -> TripPlannerState:
        llm = ChatOpenAI(temperature=0)
        
        messages = [
            SystemMessage(content="""You are a travel preference analyzer. 
            Extract key travel preferences and requirements from the conversation.
            Focus on: budget, preferred activities, travel style, must-see places, and constraints."""),
            
            HumanMessage(content=f"""
            Conversation history:
            {state.current_conversation}
            
            Extract and update travel preferences.
            """)
        ]
        
        # Process and update state
        response = llm.invoke(messages)
        # Update preferences in state
        return state
    return extract_preferences

def create_itinerary_planning_agent():
    """Agent that generates travel recommendations"""
    def plan_itinerary(state: TripPlannerState) -> TripPlannerState:
        llm = ChatOpenAI(temperature=0.7)
        
        messages = [
            SystemMessage(content="""You are a Singapore travel expert.
            Create personalized itinerary recommendations based on user preferences.
            Consider: local insights, weather, timing, and budget constraints."""),
            
            HumanMessage(content=f"""
            User Preferences:
            {state.preferences}
            
            Trip Details:
            {state.trip_details}
            
            Previous Recommendations:
            {state.recommendations}
            
            Generate updated travel recommendations.
            """)
        ]
        
        # Process and update state
        response = llm.invoke(messages)
        # Update recommendations in state
        return state
    return plan_itinerary

# 3. Create the workflow
def create_trip_planning_workflow():
    from langgraph.graph import StateGraph
    
    # Initialize workflow
    workflow = StateGraph(TripPlannerState)
    
    # Add nodes
    workflow.add_node("extract_preferences", create_preference_extraction_agent())
    workflow.add_node("plan_itinerary", create_itinerary_planning_agent())
    
    # Define edges
    workflow.add_edge("extract_preferences", "plan_itinerary")
    
    # Set entry and exit points
    workflow.set_entry_point("extract_preferences")
    workflow.set_finish_point("plan_itinerary")
    
    return workflow.compile()

# 4. FastAPI Implementation
app = FastAPI(title="Singapore Trip Planner")

class ChatInput(BaseModel):
    message: str

@app.post("/chat/")
async def chat(input: ChatInput):
    # Initialize state
    state = TripPlannerState()
    
    # Add message to conversation
    state.current_conversation.append({
        "role": "user",
        "content": input.message
    })
    
    # Run workflow
    workflow = create_trip_planning_workflow()
    final_state = workflow.invoke(state)
    
    return {
        "preferences": final_state.preferences,
        "recommendations": final_state.recommendations
    }