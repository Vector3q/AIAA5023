import re
import json

# Use symbol '\\n' if you want to indicate line breaks within the text. 
# Replace '\\"' with regular double quotes '"' around the Content string.

def complete_Writing_Prompt(topic, outline, paragraph, length):
    Writing_Prompt = f"""
        Please write a detailed paragraph for a 20,000-word article. Adhere closely to the word count requirement for each paragraph.
    
        - **Topic**: {topic}
        - **Outline**: {outline}
        - **Paragraph Number**: {paragraph}
        - **Length Requirement**: Approximately {length} words
        The output should be structured in JSON format as:
        [{{"ParagraphID": <number>, "Content": \"\"\"<string>\"\"\"}}]

        Ensure the content is a single, continuous paragraph without any segmentation. 
        

    """
    return Writing_Prompt

def complete_Planning_Prompt(topic):
    Planning_Prompt = f"""
        Please help me create the structure for a 20,000-word article. Provide a concrete idea and approximate length for each paragraph.

        - **Topic**: {topic}

        The output should be structured in JSON format as follows:
        [{{"ParagraphID": <digit>, "ParagraphIdea": \"\"\"<String>\"\"\", "ParagraphLength": <digit>}}] * n
        
        Here, 'n' represents the number of paragraphs needed.
    """
    return Planning_Prompt

def extract_json_from_text(text):
    json_pattern = re.compile(r'\[\s*{.*?}\s*\]', re.DOTALL)
    # json_text = re.sub(r"(?s)^(.*?```json\s*|\n)(.*?)(```.*)", r"\2", text).strip()
    match = json_pattern.search(text)
    # match = json_text
    

    if match:
        json_str = match.group(0)
        print(json_str)
        json_str = json_str.replace('\n', '\\n')
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            print("error text: ")
            print(text)
            print("JSON 解码错误:", e)
    else:
        print("error text: ")
        print(text)
        print("未找到 JSON 数据")

def print_article_outline(article_Outline):
    for item in article_Outline:
        print(item)

def calculate_word_count(text):
    # 使用正则表达式将文本分割成单词
    import re
    words = re.findall(r'\b\w+\b', text)
    
    # 返回单词的数量
    return len(words)
