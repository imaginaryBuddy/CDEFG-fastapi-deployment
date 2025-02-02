from fastapi import FastAPI, HTTPException
import uvicorn 
import UserInput 
app = FastAPI(title="Singapore Personality-Based Trip Planner")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Singapore Personality-Based Trip Planner"}

@app.post("/generate_itinerary")
async def generate_itinerary(user_input: str):
    try:
      request = UserInput(user_input=user_input)
      print(f"State: {state}")
      graph = create_discussion_graph()
      # logging.info(f"Graph created")
      final_state = graph.invoke(state)
      logging.info(f"Final state: {final_state}")
      print(f"Final state: {final_state}")
      print(f"Type of final state: {type(final_state)}")
      final_state_save = AgentState(**final_state)
      # redis_instance.save_game_state(request.game_id, final_state_save)
      return {'message': 'Discussion processed successfully', 'data': final_state_save.model_dump()}

    except Exception as e:
        # logging.error(f"Error processing discussion: {str(e)}")  # Log the error
        raise HTTPException(
            status_code=500,
            detail=f"Error processing discussion: {str(e)}"
        )
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

