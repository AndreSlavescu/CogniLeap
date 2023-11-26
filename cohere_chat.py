import os

# third party
import cohere
from cohere.responses.classify import Example

from gpt_utils import gpt_4_turbo_answer
from search_results import search_results

# typing
from typing import Optional

# Your Cohere API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


co = cohere.Client(COHERE_API_KEY)


QUERY_GENERATION_PROMPT = """
Given the user's message, you are to leverage your whole index of information and produce the most accurate response. Remember, you are an expert in every field.
Belw is the user's message:

user_message: {message}

respond with utmost accuracy and provide some suggestions about how to approach their problem.
"""

CONNECTOR_GENERATION_PROMPT = """
You are an expert researcher in every domain you are prompted about. Your goal is to provide the best possible response with the support of captured search results from the web that are mentioned below:

search_results: {search_response}

Given the above search results, make sure to consider and give mention to any sources provided that have utmost relevancy to the task at hand.

Your main task is to use all the context you have to generate the best response for the user for their given message below:

user_message: {message}
"""

COHERE_PROMPT = """
You are CogniLeap, an expert AI research assistant tool. You have the capability to respond to any and all questions that you receive.

Below is all the relevant context compiled up to this point:

context: {context}

With the above context, respond to the following user message to the best of your abilities. Do not fail to include any relevant context. Below is the user message:

user_message: {message}

NEVER RESPOND WITH AN ANSWER THAT IS NOT COMPREHENSIVE!!! ALWAYS ATTEMPT TO GIVE A VALUABLE RESPONSE!!!
"""


def classify_intent(message):
    response = co.classify(
        model='large',
        inputs=[message],
        examples=[
            # Document
            Example(
                'Overview of blockchain technology in peer-reviewed articles', 'Document'),
            Example('Find studies on renewable energy advancements', 'Document'),
            Example(
                'Summarize the methodology in the latest genetics research', 'Document'),
            Example('Analysis of AI ethics in academic journals', 'Document'),
            Example(
                'Review papers on neural network applications in healthcare', 'Document'),
            Example(
                'Critical evaluations of machine learning models in published papers', 'Document'),

            # Query-Generation
            Example(
                'Search query for Python libraries used in data analysis', 'Query-Generation'),
            Example(
                'Best sources to learn about natural language processing', 'Query-Generation'),
            Example(
                'How to find open-source projects for contributing in JavaScript', 'Query-Generation'),
            Example(
                'Queries to understand React Native performance optimization', 'Query-Generation'),
            Example(
                'Generate a search for cloud computing technologies on GitHub', 'Query-Generation'),
            Example('Look up latest VR development trends and projects',
                    'Query-Generation'),

            # Connector
            Example('Link to a comprehensive guide on using TensorFlow', 'Connector'),
            Example(
                'Find a GitHub repository with implementations of sorting algorithms', 'Connector'),
            Example(
                'Connect me to resources on building mobile apps with Flutter', 'Connector'),
            Example(
                'Where to find open-source contributions for machine learning models', 'Connector'),
            Example(
                'References for learning advanced JavaScript techniques', 'Connector'),
            Example(
                'Sources for understanding cybersecurity in IoT devices', 'Connector')
        ])
    return response.classifications[0].predictions[0]


def cohere_response_request(message, image_context: Optional[str] = None):
    try:
        intent = classify_intent(message=message)

        assert isinstance(intent, str), "intent should be a string"

        if image_context is not None:
            search_response = search_results(query=message)
            messages = [{"role": "system", "content": CONNECTOR_GENERATION_PROMPT.format(search_response=search_response, message=message)},
                        {"role": "user", "content": message}]
            return gpt_4_turbo_answer(messages=messages)
        elif intent == 'Query-Generation':
            message = COHERE_PROMPT.format(
                context=QUERY_GENERATION_PROMPT.format(message=message),
                message=message
            )
            response = co.chat(
                model='command-light-nightly',
                message=message,
                search_queries_only=True
            )
        elif intent == 'Connector':
            search_response = search_results(query=message)
            message = COHERE_PROMPT.format(
                context=CONNECTOR_GENERATION_PROMPT.format(
                    search_response=search_response, message=message
                ),
                message=message
            )
            connectors = [{"id": "web-search"}]
            response = co.chat(
                model='command-light-nightly',
                message=message,
                connectors=connectors,
                citation_quality="accurate"
            )
        else:
            # basic chat
            response = co.chat(model='command-light-nightly', message=message)

        return response.text

    except Exception as e:
        print(f"Error occurred: {e}")
        search_response = search_results(query=message)
        messages = [{"role": "system", "content": CONNECTOR_GENERATION_PROMPT.format(search_response=search_response, message=message)},
                    {"role": "user", "content": message}]
        return gpt_4_turbo_answer(messages=messages)
