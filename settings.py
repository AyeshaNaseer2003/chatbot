from pydantic import BaseSettings

class Settings(BaseSettings):
 
    GEMINI_API_KEY: str
    TAVILY_API_KEY: str
    MONGO_URI: str

    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str

    class Config:
        env_file = ".env"

settings = Settings()
