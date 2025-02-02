from pydantic import BaseModel, Field
from typing import Optional
from typing import List

class UserInput(BaseModel):
    """
    User's Input
    """
    user_input: str = Field(description="User's input")

class PersonalityProfile(BaseModel):
    """
    Personality Profile that the Information Extraction Agent will extract from the user's input
    """
    personality_traits: List[str] = Field(
        description="Personality traits (e.g., 'extroverted', 'adventurous', 'organized')"
    )
    interests: List[str] = Field(
        description="Specific interests (e.g., 'photography', 'history', 'food')"
    )
    energy_level: str = Field(
        description="Preferred activity intensity (Low/Medium/High)"
    )
    social_preference: str = Field(
        description="Solo/Small Group/Large Group"
    )
    budget_level: str = Field(
        description="$ (0-20), $$ (20-100), $$$ (100+)"
    )

class Activity(BaseModel):
    """
    Activity that the Activity Recommendation Agent will recommend
    This is also used in the "database" of activities 
    """
    name: str
    location: str
    personality_match: List[str] = Field(
        description="Personality traits this activity suits"
    )
    social_level: str
    energy_required: str
    price_range: str
    best_for: List[str] = Field(
        description="Perfect for people who love..."
    )
class ActivitiesAndReason(BaseModel):
    """
    Activities and Reason for Recommendations that the Activity Recommendation Agent will recommend
    """
    activities: List[Activity] = Field(description="Activities")
    reason: str = Field(description="Reason for recommendations")

# Can inherit classes 
class ActivityAndTime(Activity):
    """
    Activity and Time of the activity that the Activity Recommendation Agent will recommend
    """
    time: str = Field(description="Time of the activity")

class Itinerary(BaseModel):
    """
    Itinerary of the trip that the Trip Planner Agent will recommend
    """
    day: str = Field(description="Day of the trip")
    activities: List[ActivityAndTime] = Field(description="Activities with the time for the day's itinerary")

class TripPlan(BaseModel):
    """
    Trip Plan that the Trip Planner Agent will recommend
    """
    itinerary: List[Itinerary] = Field(description="List of daily itineraries")

class AgentState(BaseModel):
    """
    State of the Agent
    """
    user_input: UserInput = Field(description="User's input")
    personality_profile: Optional[PersonalityProfile] = Field(default=None, description="Personality profile")
    activity_recommendations: Optional[List[Activity]] = Field(default=None, description="Activity recommendations")
    reason_for_recommendations: Optional[str] = Field(default=None, description="Reason for recommendations")
    final_trip_plan: Optional[TripPlan] = Field(default=None, description="Final trip plan")
