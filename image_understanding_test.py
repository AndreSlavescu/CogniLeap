from image_understanding import ImageUnderstanding
import time


def runner(image_path: str, user_query: str) -> str:
    img_understanding = ImageUnderstanding(
        path=image_path
    )

    return img_understanding.final_master_prompt(user_query=user_query)


if __name__ == '__main__':
    start = time.time()
    result = runner(
        image_path="temp_images/Screenshot 2023-11-18 at 12.33.48 PM.png",
        user_query="What are some other examples that look like the following image?"
    )
    end = time.time()

    print("Result: ", result)
    print("Time: ", end - start)
