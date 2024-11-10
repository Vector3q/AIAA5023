from zhipuai import ZhipuAI
from Judge import text_Judgement, complete_Judge_Prompt
import utils

times = 0
client = ZhipuAI(api_key="f36c12ddada4a0db960a6c2c926f7b3c.jnit4yENhz8vZW9d")  # Please fill in your own APIKey

# The maximum number of tokens for model output, maximum output is 4095, default value is 1024.

def text_generation(prompt, times):
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
    print("total word count: "+ str(utils.calculate_word_count(response.choices[0].message.content)))
    print("############################################################################################")

    return response.choices[0].message, times


############################################################################################################
# The following functions are for you to implement
# You can implement functions for saving and loading files, and planning
# You can also implement functions for the chatbot to interact with the user
############################################################################################################

def savig_files(results):
    # to save to generated_text.txt
    with open("50022955.txt", "w") as f:
        f.write(results)

def load_files():
    pass

def planning(topic, times):

    results, times = text_generation(utils.complete_Planning_Prompt(topic), times)
    article_Outline = utils.extract_json_from_text(results.content)
    print("Article Outline: ")
    print("") 
    utils.print_article_outline(article_Outline)

    initial_article = ""
    for item in article_Outline:
        results, times = text_generation(utils.complete_Writing_Prompt(topic, str(article_Outline), item['ParagraphIdea'], item['ParagraphLength']), times)
        results = utils.extract_json_from_text(results.content)
        print(str(results[0]['ParagraphID']))

        _paragraph = str(results[0]['Content']) + "\n\n"
        print(_paragraph)
        initial_article += str(_paragraph)

    
    print("############################################################################################")

    savig_files(initial_article)
    return times

def others():
    pass

# Simple example
topic = "Application of Nanotechnology in Medicine"


print("############################################################################################")
times = planning(topic, times)


print("API calls:", times)
print("############################################################################################")


# results, times = text_generation("generating a 50000-words long English text with the topic of: " + topic, times)
# results, times = text_generation("polish the text and extend each part with more content, " + results.content, times)

# savig_files(results.content)



# Judgement

# score, times = text_Judgement(topic, results.content, times)
# print(score)


############################################################################################################

