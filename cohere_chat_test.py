from image_understanding import ImageUnderstanding
from cohere_chat import cohere_response_request
import time


def runner_no_image(user_query: str) -> str:
    return cohere_response_request(message=user_query)


def runner_image(image_path: str, user_query: str) -> str:
    img_understanding = ImageUnderstanding(
        path=image_path
    )

    final_context = img_understanding.final_master_prompt(
        user_query=user_query
    )

    return cohere_response_request(message=user_query, image_context=final_context)


if __name__ == '__main__':
    start_no_image = time.time()
    result_no_image = runner_no_image(
        user_query="How can I replicate UI similar to how google's front page looks like?"
    )
    end_no_image = time.time()
    print("Result for no Image: ", result_no_image)
    print("Time for no Image: ", end_no_image - start_no_image)

    start_image = time.time()
    result_image = runner_image(
        image_path="temp_images/Screenshot 2023-11-18 at 12.33.48 PM.png",
        user_query="How can I replicate UI similar to the one in the image?"
    )
    end_image = time.time()
    print("Result for Image: ", result_image)
    print("Time for Image: ", end_image - start_image)
