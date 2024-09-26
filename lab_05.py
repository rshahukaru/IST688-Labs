import streamlit as st
import requests
import openai
import json

st.title("Lab 5 - Weather and Clothing Suggestion Bot")

# Read API keys from st.secrets
openai_api_key = st.secrets["openai_api_key"]
openweathermap_api_key = st.secrets["openweathermap_api_key"]

# Set OpenAI API key
openai.api_key = openai_api_key

# Define function to get current weather
def get_current_weather(location, API_KEY):
    if "," in location:
        location = location.split(",")[0].strip()
    urlbase = "https://api.openweathermap.org/data/2.5/"
    urlweather = f"weather?q={location}&appid={API_KEY}"
    url = urlbase + urlweather
    response = requests.get(url)
    data = response.json()
    if data.get("cod") != 200:
        return {"error": data.get("message", "Error retrieving weather data")}
    # Extract temperatures & Convert Kelvin to Celsius
    temp = data['main']['temp'] - 273.15
    feels_like = data['main']['feels_like'] - 273.15
    temp_min = data['main']['temp_min'] - 273.15
    temp_max = data['main']['temp_max'] - 273.15
    humidity = data['main']['humidity']
    weather_description = data['weather'][0]['description']
    return {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "humidity": round(humidity, 2),
        "description": weather_description
    }

# Define function descriptions for OpenAI Function Calling
function_descriptions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. 'Syracuse, NY'",
                },
            },
            "required": ["location"],
        },
    }
]

# Prompt the user for input
user_input = st.text_input("Ask something about the weather and clothing:")

if user_input:
    # Step 1: Send the user's message to OpenAI, with function definitions
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides weather information and clothing suggestions, and advice on whether it's a good day for a picnic."},
        {"role": "user", "content": user_input}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=function_descriptions,
        function_call="auto",
    )
    
    response_message = response['choices'][0]['message']
    
    # Step 2: Check if the assistant wants to call a function
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_args = response_message["function_call"]["arguments"]
        # Parse function arguments
        try:
            function_args = json.loads(function_args)
        except json.JSONDecodeError:
            function_args = {}
        location = function_args.get("location")
        if not location:
            # If no location provided, use "Syracuse, NY" as default
            location = "Syracuse, NY"
        # Execute the function
        weather_info = get_current_weather(location, openweathermap_api_key)
        if "error" in weather_info:
            # If there was an error getting the weather
            assistant_message = f"Sorry, I couldn't get the weather for {location}. {weather_info['error']}"
            st.write(assistant_message)
        else:
            # Convert weather_info to a string to send back to the assistant
            weather_info_str = json.dumps(weather_info)
            
            # Step 3: Send the assistant's response and function result back to OpenAI
            messages.append(response_message)
            messages.append({
                "role": "function",
                "name": function_name,
                "content": weather_info_str,
            })
            
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
            )
            
            assistant_message = second_response['choices'][0]['message']['content']
            
            # Display the assistant's final answer
            st.write(assistant_message)
    else:
        # Assistant didn't call any function, just display the response
        assistant_message = response_message['content']
        st.write(assistant_message)
