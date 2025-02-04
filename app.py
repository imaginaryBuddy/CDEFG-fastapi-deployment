from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 
from pydantic import BaseModel, Field
from trip_planner import create_trip_planner_graph
from utils import AgentState, UserInput


app = FastAPI(title="Singapore Personality-Based Trip Planner")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this frontend's origin
    allow_credentials=False,
    allow_methods=["POST", "GET"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Singapore Personality-Based Trip Planner"}

@app.post("/generate_itinerary", description="Enter a user input that describes the following: \n 1. Your personality traits (e.g., 'extroverted', 'adventurous', 'organized') \n 2. Your budget ($ (0-20), $$ (20-100), $$$ (100+)) \n 3. Your interests (e.g., 'photography', 'history', 'food')   \n 4. Your energy level (high, medium, low) \n 5. Your social preference (solo, small group, large group)")
async def generate_itinerary(user_input: UserInput):
    try:
      graph = create_trip_planner_graph()
      # create an AgentState object to pass to the graph 
      state = AgentState(user_input=user_input) 
      # invoke the graph 
      final_state = graph.invoke(state)

      print(f"Final state: {final_state}")
      print(f"Type of final state: {type(final_state)}")

      final_state_save = AgentState(**final_state)
      final_trip_plan = final_state_save.final_trip_plan or None
      reason_for_recommendations = final_state_save.reason_for_recommendations or None
      # model_dump() converts the final state to a dictionary, so that it will be JSON serializable by FastAPI
      return {'message': 'Trip plan generated successfully', 
              'final_trip_plan': final_trip_plan, 
              'reason_for_recommendations': reason_for_recommendations if final_state_save.reason_for_recommendations else None
              }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing discussion: {str(e)}"
        )
    

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

