from langfuse import get_client
lf = get_client()

# Create or update a production meta‑prompt in Langfuse
lf.create_prompt(
  name="my_gemini_meta",
  type="chat",
  prompt=[
    {"role":"system","content":"You are a helpful assistant. Always use TavilySearch for current info."},
    {"role":"user","content":"{{user_input}}"}
  ],
  labels=["production"],
  config={"model":"gemini-2.0-flash"}
)
