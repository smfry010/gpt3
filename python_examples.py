#python -m venv base

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_function(text, **kwargs):  
    TEMP = kwargs.get('temperature', .7)
    MAX_TOKENS = kwargs.get('max_tokens', 256)
    TOP_P = kwargs.get('top_p', 1)
    FREQ_PENALTY = kwargs.get('frequency_penalty', 0)
    PRES_PENALTY = kwargs.get('presence_penalty', 0)

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=text,
        temperature=TEMP,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQ_PENALTY,
        presence_penalty=PRES_PENALTY
    )
    return response['choices'][0]['text']

# Q&A
gpt_function("What party did Donald Trump belong to?", temperature=0, max_tokens=50)
gpt_function("What is Donald Trump's middle name?", temperature=0, max_tokens=50)
gpt_function("How famous was Goldie Hawn?", temperature=.7)

# summarize text
long_text = "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus."
gpt_function(long_text, temperature=.1, max_tokens=100)

# translate text
translate_text = "Translate this into 1. French, 2. Spanish, 3. German, and 4.Japanese:\n\nWhat rooms do you have available?\n\n1."
gpt_function(translate_text, temperature=0)

# chatbot
gpt_function("Who are you voting for in this election?", temperature=1)

# sarcastic chatbot
gpt_function("Marv is a sarcastic chatbot.\n\n \
    You: What time is it in Italy?\n \
    Marv: Really? You can't check Google?\n \
    You: Why are you so cranky?\n \
    Marv: If you would ask an intelligent question, I would be happy\n \
    You: Can you help me with my homework?\n \
    Marv:")

# dark chatbot
gpt_function("Demon is a dark chatbot.\n\n \
    You: When is it best to walk your dog?\n \
    Demon: Dogs are food, they should not be walked.\n \
    You: Why are you so cranky?\n \
    Demon: My focus on the negative, that is all.\n \
    You: Are babies sweet?\n \
    Demon:")

# correct incorrect grammar and spelling
grammar_text = "Lauren no went to the market."
gpt_function(grammar_text, temperature=0, max_tokens=60)

# find the pattern
baseball_text1 = "Provide popular baseball player names from the provided team:\n \
    Team: St Louis Cardinals\n \
    Players: Lou Brock, Stan Musial\n \
    Team: Cubs\n \
    Players:"
gpt_function(baseball_text1)

baseball_text2 = "Provide the baseball team from the provided names:\n \
    Name: Lou Brock, Stan Musial\n \
    Team: Cardinals\n \
    Name: Barry Bonds, Kevin Mitchell\n \
    Team:"
gpt_function(baseball_text2)

# convert notes into summary
notes_text = "Convert my short hand into a first-hand account of the meeting:\n \
    Tom: Profits up 50%\n \
    Jane: New servers are online\n \
    Kjel: Need more time to fix software\n \
    Jane: Happy to help\n \
    Parkman: Beta testing almost done"
gpt_function(notes_text)

# create interview questions
interview_text = "Create a list of 8 questions for my interview for a manager of data science."
gpt_function(interview_text, temperature=.8)

# sentiment scoring
sentiment_text = "Classify the sentiment in these tweets:\n \
    1. I can't stand homework\n \
    2. This sucks. I'm bored 😠\n \
    3. I can't wait for Halloween!!!\n \
    4. My cat is adorable ❤️❤️\n \
    5. I hate chocolate\n \
    Tweet sentiment ratings:"
gpt_function(sentiment_text, max_tokens=60, temperature=0)