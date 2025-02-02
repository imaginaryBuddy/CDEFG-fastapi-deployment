from utils import UserInput, AgentState
from trip_planner import create_trip_planner_graph
from typing import List
from pydantic import BaseModel, Field



sample_data = UserInput(user_input="""I am an extrovert, 
                        I like history and food, 
                        I am medium energy, 
                        I prefer small groups, 
                        and I have a budget of $200""")


graph = create_trip_planner_graph()

state = AgentState(user_input=sample_data)

final_state = graph.invoke(state)
final_state_save = AgentState(**final_state)
print(final_state_save.final_trip_plan.model_dump())
