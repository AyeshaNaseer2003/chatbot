# mongo_utils.py
import motor.motor_asyncio
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.chatbot_db
messages_collection = db.messages

async def save_message(user: str, message: str, response: str):
    try:
        result = await messages_collection.insert_one({
            "user": user,
            "message": message,
            "response": response
        })
        print(f"Inserted with ID: {result.inserted_id}")
    except Exception as e:
        print("Error saving message:", e)

async def get_history(user: str):
    cursor = messages_collection.find({"user": user})
    return await cursor.to_list(length=100)  # limit for performance


