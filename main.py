from utils.stringify_topics import stringify_topics
from chatbots.OllamaChatBot import OllamaChatBot

#the user will be able to change the model
model = "gemma3:4b"

#configuration... later will be fully customizable
base_language = "Portuguese"
wanted_language = "German"
proficiency = ["basic", "moderate", "advanced", "fluent"]
topics = ["Programming", "Tech", "UX", "Accessibility"]

model = OllamaChatBot(model, base_language, wanted_language, proficiency[0], stringify_topics(topics))

while True:
    msg = input("msg: ")
    res = model.generateAnswer(msg)
    print(res)