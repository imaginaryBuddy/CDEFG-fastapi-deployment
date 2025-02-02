from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from utils import AgentState, PersonalityProfile, Activity, ActivitiesAndReason, TripPlan
import os
from dotenv import load_dotenv
 
load_dotenv(override=True)

# Sample activity database
ACTIVITIES_DB = [
    Activity(
        name="Indoor Skydiving at iFly Singapore",
        location="Sentosa",
        personality_match=["adventurous", "thrill-seeker", "energetic"],
        social_level="Solo/Small Group",
        energy_required="High",
        price_range="$$$",
        best_for=["adrenaline rush", "unique experiences"]
    ),
    Activity(
        name="Bungee Jumping at AJ Hackett Sentosa",
        location="Sentosa",
        personality_match=["adventurous", "daredevil", "energetic"],
        social_level="Solo/Small Group",
        energy_required="High",
        price_range="$$$",
        best_for=["adrenaline rush", "extreme sports"]
    ),
    Activity(
        name="Hiking at MacRitchie Reservoir & TreeTop Walk",
        location="MacRitchie Reservoir",
        personality_match=["nature-loving", "adventurous", "outdoorsy"],
        social_level="Solo/Small Group",
        energy_required="Medium-High",
        price_range="$",
        best_for=["hiking", "nature", "outdoor exploration"]
    ),
    Activity(
        name="Cycling at East Coast Park",
        location="East Coast Park",
        personality_match=["outdoorsy", "fitness-oriented", "adventurous"],
        social_level="Solo/Small Group",
        energy_required="Medium",
        price_range="$",
        best_for=["cycling", "scenic views", "beachside activities"]
    ),
    Activity(
        name="Kayaking in Mandai Mangrove",
        location="Mandai",
        personality_match=["adventurous", "nature-loving", "curious"],
        social_level="Small Group",
        energy_required="Medium",
        price_range="$$",
        best_for=["water activities", "wildlife spotting", "eco-tourism"]
    ),
    Activity(
        name="Gardens by the Bay (Floral Fantasy & Cloud Forest)",
        location="Marina Bay",
        personality_match=["nature-loving", "relaxed", "curious"],
        social_level="Solo/Small Group",
        energy_required="Low-Medium",
        price_range="$$",
        best_for=["botanical experience", "photography", "unique architecture"]
    ),
    Activity(
        name="National Gallery Singapore",
        location="City Hall",
        personality_match=["artistic", "introspective", "curious"],
        social_level="Solo/Small Group",
        energy_required="Low",
        price_range="$$",
        best_for=["art appreciation", "museum exploration", "history"]
    ),
    Activity(
        name="Peranakan Museum Tour",
        location="Armenian Street",
        personality_match=["history buff", "cultural enthusiast", "curious"],
        social_level="Solo/Small Group",
        energy_required="Low",
        price_range="$$",
        best_for=["heritage", "culture", "educational tours"]
    ),
    Activity(
        name="Hawker Centre Food Tour",
        location="Various Locations (Maxwell, Tiong Bahru, Old Airport Road)",
        personality_match=["foodie", "explorer", "social"],
        social_level="Solo/Small Group",
        energy_required="Low",
        price_range="$",
        best_for=["local food", "budget-friendly", "cultural experience"]
    ),
    Activity(
        name="Cafe-Hopping in Tiong Bahru",
        location="Tiong Bahru",
        personality_match=["foodie", "aesthetic lover", "creative"],
        social_level="Solo/Small Group",
        energy_required="Low",
        price_range="$$",
        best_for=["brunch", "coffee lovers", "trendy spots"]
    ),
    Activity(
        name="Luxury Shopping at Marina Bay Sands",
        location="Marina Bay Sands",
        personality_match=["fashion-forward", "luxury seeker", "shopaholic"],
        social_level="Solo/Small Group",
        energy_required="Medium",
        price_range="$$$$",
        best_for=["high-end fashion", "luxury experience", "exclusive brands"]
    ),
    Activity(
        name="Sunset Cruise along Marina Bay",
        location="Marina Bay",
        personality_match=["romantic", "relaxed", "nature-loving"],
        social_level="Small Group/Couples",
        energy_required="Low",
        price_range="$$$",
        best_for=["scenic experience", "luxury", "unique evening activity"]
    ),
    Activity(
        name="Clarke Quay Nightlife & Bar-Hopping",
        location="Clarke Quay",
        personality_match=["social", "party-lover", "energetic"],
        social_level="Small/Large Group",
        energy_required="High",
        price_range="$$$",
        best_for=["bars", "nightlife", "music"]
    ),
    Activity(
        name="Neon Bowling at K Bowling Club",
        location="Somerset",
        personality_match=["social", "fun-loving", "casual"],
        social_level="Small/Large Group",
        energy_required="Medium",
        price_range="$$",
        best_for=["group activity", "bowling", "casual fun"]
    ),
    Activity(
        name="Comedy Night at The Merry Lion",
        location="River Valley",
        personality_match=["humorous", "social", "casual"],
        social_level="Small Group",
        energy_required="Medium",
        price_range="$$",
        best_for=["stand-up comedy", "entertainment", "night out"]
    )
]


def create_information_extraction_agent():
    def extract_information(state: AgentState) -> AgentState:
        llm_information_extraction = AzureChatOpenAI(
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
            model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
            api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            temperature=0.2,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
        )

        messages = [
            SystemMessage(content=f"""You are a Singapore trip planner assistant. You are given a user's input and you need to extract the information from the user's input.
                You need to extract the following information:
                - Personality traits
                - Interests
                - Energy level
                - Social preference
                - Budget level
                          
                If the user's input is not clear, please generate a default personality profile.
            """), 
            HumanMessage(content=f"""
            User's input: {state.user_input}
            """)
        ]

        llm_structured_output_information_extraction = llm_information_extraction.with_structured_output(PersonalityProfile)
        response = llm_structured_output_information_extraction.invoke(messages)
        state.personality_profile = response
        return state
    return extract_information

def create_searching_agent():
    def search_activities(state: AgentState) -> AgentState:
        llm_search = AzureChatOpenAI(
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
            model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
            api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            temperature=0.2,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
        )

        messages = [
            SystemMessage(content=f"""You are a Singapore trip planner assistant. You are given a list of possible destinations in Singapore and the personality profile
                          of the user. You need to recommend activities suitable for the user to do. If you deem that there are other acitvities available and suitable for
                          the user that's not in the database, feel free to sugges them.
                          You need to output a list of destinations that the user can visit. For each destination, you need to output the following information:
                          - name of destination
                          - location of destination 
                          - personality traits that the destination suits 
                          - social level that the destination suits 
                          - energy level that the destination suits 
                          - budget level that the destination suits 
                          - activities that the destination has 

                          The database of activities is as follows: {ACTIVITIES_DB}

                          Please also output a short and concise reason why you chose the activities and destinations for the user based on their personality.
                           """)
        ]

        llm_structured_output_search = llm_search.with_structured_output(ActivitiesAndReason)
        response = llm_structured_output_search.invoke(messages)
        state.activity_recommendations = response.activities
        state.reason_for_recommendations = response.reason
        return state 
    return search_activities

def create_itinerary_agent():
    def generate_itinerary(state: AgentState) -> AgentState:
        llm_recommend = AzureChatOpenAI(
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
            model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
            api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            temperature=0.6,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
        )

        messages = [
            SystemMessage(content="""You are a Singapore 2 day 1 night trip planner assistant who is given a list of activities that the user can visit.
                          You need to construct a well-paced itinerary for the user to follow, taking into account 
                          the travel time and the distances between destinations. 
                          You will need to extract the following information: 
                          - Day n, where n is the day of the trip
                          - Activities for the day 
                          - Time of the activities 
                          - Location of the activities 
                          - Personality traits that the activities suits 
                          - Social level that the activities suits 
                          - Energy level that the activities suits 
                          - Budget level that the activities suits 

                            Keep as concise as possible
                           """),
            HumanMessage(content=f"""
            Activities: {state.activity_recommendations}
            Reason for recommendations: {state.reason_for_recommendations}
            """)
        ]

        llm_structured_output_itinerary = llm_recommend.with_structured_output(TripPlan)
        response = llm_structured_output_itinerary.invoke(messages)
        state.final_trip_plan = response
        return state
    
    return generate_itinerary

def create_trip_planner_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_information", create_information_extraction_agent())
    workflow.add_node("search_activities", create_searching_agent())
    workflow.add_node("generate_itinerary", create_itinerary_agent())
    
    #define edges
    workflow.add_edge("extract_information", "search_activities")
    workflow.add_edge("search_activities", "generate_itinerary")
    
    # Set entry point
    workflow.set_entry_point("extract_information")
    
    # Set exit point
    workflow.set_finish_point("generate_itinerary")
    
    return workflow.compile()

