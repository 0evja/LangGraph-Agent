import os
import openai
from dotenv import load_dotenv, find_dotenv

# 获取openai API_KEYS
def get_openai_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['OPENAI_API_KEY']

# 获取base_url
def get_base_url():
    _ = load_dotenv(find_dotenv())
    return os.environ['BASE_URL']

# 获取tavily API
def get_tavily_api():
    _ = load_dotenv(find_dotenv())
    return os.environ['TAVILY_API']
