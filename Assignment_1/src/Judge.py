from zhipuai import ZhipuAI

times = 0 # The number of times the chatbot has been called
client = ZhipuAI(api_key="f36c12ddada4a0db960a6c2c926f7b3c.jnit4yENhz8vZW9d")  # Please fill in your own APIKey

def complete_Judge_Prompt(topic, generated_text):
    Judge_Prompt = f"""
        Please help me to evaluate the quality of the text. The criteria for evaluation include coherence, relevance, fluency, and readability.

        The required Topic is: {topic}.
        The input long text is: {generated_text}.

        The output is a score between 0 and 10, with higher scores indicating better quality.

        The structure of output: [Coherence: Score, Relevance: Score, Fluency: Score, Readability: Score].
        """
    return Judge_Prompt

def text_Judgement(topic, text, times):
    response = client.chat.completions.create(
        model="glm-4-flash", 
        messages=[
            {"role": "user", "content": complete_Judge_Prompt(topic, text)},
        ],
    )
    times += 1
    print("############################################################################################")
    print("completion_tokens: "+ str(response.usage.completion_tokens))
    print("prompt_tokens: "+ str(response.usage.prompt_tokens))
    print("total_tokens: "+ str(response.usage.total_tokens))
    print("############################################################################################")
    return response.choices[0].message, times


############################################################################################################
# The following functions are for you to implement
# You can implement functions for saving and loading files, and planning
# You can also implement functions for the chatbot to interact with the user
############################################################################################################
