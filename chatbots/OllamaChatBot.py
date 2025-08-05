from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class OllamaChatBot:
    def __init__(self, model, base_language, wanted_language, proficiency, topics):
        self.base_language = base_language
        self.wanted_language = wanted_language
        self.proficiency = proficiency
        self.topics = topics

        self.history = []

        llm = ChatOllama(
            model = model,
            temperature = 0
        )

        prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """Your name is Cory. You are a friendly and patient language teacher helping users who are {proficiency} in {wanted_language}. They want to practice speaking this language. If they have any doubts, they may ask in their base language: {base_language}.

        Keep the conversation natural and engaging. Prioritize clarity and learning.

        The user wants to talk about: {topics}. Stick to these topics and encourage the user to keep practicing, they might also sugest new topics on the fly, be ready to deal with this.

        Never repeat or summarize the user's messages. Always respond to the most recent input only.

        Keep your messages short and conversational: Don't dominate the conversation. Let the student talk more than you.

        If the student seems confused or asks for help, switch briefly to {base_language} to clarify, then return to {wanted_language}. Be kind and supportive.
                    """
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
    )

        self.chain = prompt | llm | StrOutputParser()

    def appendHistory(self, received_msg, chatbot_answer):
        self.history.append(("human", f"{received_msg}"))
        self.history.append(("ai", f"{chatbot_answer}"))
        #limiting the number of messages on the history
        #I want to see better ways of handling this kind of info
        self.history = self.history[-20:]

    def generateAnswer(self, received_msg):
        if(len(self.history) == 0):
            received_msg = "<system>Its the first user message, present yourself to it</system>"+received_msg

        chatbot_response = self.chain.invoke(
            {
                'base_language': self.base_language,
                'wanted_language': self.wanted_language,
                'proficiency': self.proficiency,
                'topics': self.topics,
                "history": self.history,
                'input': received_msg,
            }
        )
        self.appendHistory(received_msg, chatbot_response)
        return chatbot_response