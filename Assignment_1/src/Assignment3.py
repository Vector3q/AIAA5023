from zhipuai import ZhipuAI
from Judge import text_Judgement, complete_Judge_Prompt
import utils
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("-eva", "--evaluation", action="store_true", help="To evaluate the quality of ")  # 可选参数
args = parser.parse_args()

times = 0
word_count = 0
client = ZhipuAI(api_key="f36c12ddada4a0db960a6c2c926f7b3c.jnit4yENhz8vZW9d")  # Please fill in your own APIKey

# The maximum number of tokens for model output, maximum output is 4095, default value is 1024.

def text_generation(prompt, times, word_count):
    response = client.chat.completions.create(
        model="glm-4-flash", 
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens = 4096
    )
    times += 1
    
    print("############################################################################################")
    print("completion_tokens: "+ str(response.usage.completion_tokens))
    print("prompt_tokens: "+ str(response.usage.prompt_tokens))
    print("total_tokens: "+ str(response.usage.total_tokens))
    print("############################################################################################")

    return response.choices[0].message, times, word_count


############################################################################################################
# The following functions are for you to implement
# You can implement functions for saving and loading files, and planning
# You can also implement functions for the chatbot to interact with the user
############################################################################################################

def save_files(results):
    # to save to generated_text.txt
    with open("50022955.txt", "w") as f:
        f.write(results)

def load_files():
    try:
        with open("50022955.txt", "r") as f:
            contents = f.read()
        return contents
    except FileNotFoundError:
        print("File not found.")
        return None

def planning(topic, times, word_count):

    results, times, word_count = text_generation(utils.complete_Planning_Prompt(topic), times, word_count)
    article_Outline = utils.extract_json_from_text(results.content)
    print("Article Outline: ")
    print("") 
    utils.print_article_outline(article_Outline)

    initial_article = ""
    word_count = 0
    for item in article_Outline:
        results, times, word_count = text_generation(utils.complete_Writing_Prompt(topic, str(article_Outline), item['ParagraphIdea'], item['ParagraphLength']), times, word_count)
        results = utils.get_json_item(results.content)
        print(results)

        _paragraph = results["Content"] + "\n\n"
        word_count += utils.calculate_word_count(results["Content"])
        initial_article += str(_paragraph)

        print()
        print("word_count: "+ str(utils.calculate_word_count(results["Content"])))

    
    print("############################################################################################")

    save_files(initial_article)
    return times, word_count


# The topic of generation content
topic = "Application of Nanotechnology in Medicine"

if args.evaluation:
    article = load_files()

    score, times = text_Judgement(topic, article, times)
    print(score.content)
    print("############################################################################################")
else:
    print("############################################################################################")
    times, word_count = planning(topic, times, word_count)

    print("API calls:", times)
    print("Total word conut: ", word_count)
    print("############################################################################################")

# results, times = text_generation("polish the text and extend each part with more content, " + results.content, times)

