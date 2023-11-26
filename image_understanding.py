import torch
import clip
from PIL import Image

from gpt_utils import gpt_4_turbo_answer
from search_results import search_results

SEARCH_PROMPT = """
Below are the search results: 

search_results: {search_results}

You are to do a further ranking process of these results based on the semantic similarity to the following context:

augmented_context: {augmented_context}

You are to construct a final master prompt to drive action. This master prompt should be the context above in addition to the search results, in addition to a routing mechanism.
Below is a key example:

<!--> IMPORTANT CONTEXT <--!>
The user is asking about the particular structure of how to replicate the UI design of the mentioned image.
The webpage seems to be a detailed and well structured webpage for conversing with an AI chatbot.
You are to provide some insight about possible best fitting results that are retrieved.

Below are the retrieved results:

search_results: "associated search results"

Given all of the above, you are to provide the best possible response and direction to lead the user ot their desired goal.
<!--> IMPORTANT CONTEXT <--!>
"""


STRUCTURE_CONTEXT = """
<!--> IMPORTANT CONTEXT <--!>
Given a description of an image that you are to assume is completely accurate, derive appropriate context that represents the image to perfection.
Below is the description of the inputted image:

description: {description}

And the associated query from the user:

user_query: {user_query}
<!--> IMPORTANT CONTEXT <--!>

<!--> IMPORTANT CONTEXT <--!>
Below is an example of how this might be done for some input:

user_query: How can I replicate the UI for the following chat based application?

description: a webpage for conversing with an AI driven chatbot

the following context derived is the response:

The user is asking about the particular structure of how to replicate the UI design of the mentioned image.
The webpage seems to be a detailed and well structured webpage for conversing with an AI chatbot.
You are to provide some insight about possible best fitting results that are retrieved.

Below are the retrieved results:
<!--> IMPORTANT CONTEXT <--!>

Make sure to try to match at the best of your ability the above example with the appropriate derived context.
"""


class ImageUnderstanding:
    def __init__(self, path) -> None:
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # image path
        self.path = path

        # Initialize CLIP model with the appropriate device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def _process_image_with_clip(self, image_path: str) -> str:
        image = self.preprocess(
            Image.open(image_path)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)

        # Labels
        descriptions = [
            "a research document for AI",
            "a research document for Computer Systems",
            "a research document for programming",
            "a research document for machine learning",
            "a research document for data science",
            "a research document for cybersecurity",
            "a research document for quantum computing",
            "a github repo for an AI application",
            "a github repo for a web application",
            "a github repo for a mobile application",
            "a github repo for game development",
            "a github repo for data analysis tools",
            "a github repo for machine learning models",
            "a webpage for conversing with an AI driven chatbot",
            "a webpage for conversing with people",
            "a webpage for an API",
            "a webpage for e-commerce",
            "a webpage for educational content",
            "a webpage for tech news",
            "a webpage for sports updates",
            "a webpage for health and wellness"
        ]

        text_tokens = clip.tokenize(descriptions).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)

        similarities = (image_features @ text_features.T).softmax(dim=-1)
        best_match_index = similarities.argmax(dim=-1).item()
        best_description = descriptions[best_match_index]

        return best_description

    def _structure_context(self, user_query: str) -> str:
        """
        Build's up relevant context given a user query and the associated description of an image using HyDE

        Requires:
            type(user_query) is str
            type(description) is str

        Parameters:
            user_query: str, query that user inputs that associates to the image

        Returns:
            str
        """
        assert isinstance(user_query, str), "user query must be a string."

        description = self._process_image_with_clip(image_path=self.path)
        assert isinstance(description, str), "description must be a string."

        messages = [
            {
                "role": "system",
                "content": STRUCTURE_CONTEXT.format(
                    description=description, user_query=user_query
                )
            }
        ]

        response = gpt_4_turbo_answer(
            messages=messages
        )

        return response

    def final_master_prompt(self, user_query: str) -> str:
        assert isinstance(user_query, str), "user query must be a string."

        augmented_context = self._structure_context(user_query=user_query)
        assert isinstance(
            augmented_context,
            str
        ), "augmented context must be a string"

        messages = [
            {
                "role": "system",
                "content": SEARCH_PROMPT.format(
                    search_results=search_results(query=augmented_context),
                    augmented_context=augmented_context
                )
            }
        ]

        response = gpt_4_turbo_answer(
            messages=messages
        )

        return response
