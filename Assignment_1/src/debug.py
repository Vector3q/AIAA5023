input = '''```json
[
    {
        "ParagraphID": 2,
        "Content": "The journey of nanotechnology in medicine is a testament to the relentless pursuit of scientific discovery and its transformative potential. Originating from the Greek word "nanos," meaning "dwarf," the term nanotechnology refers to the manipulation and control of matter at the nanoscale. The concept of nanotechnology has its roots in the early 20th century when scientists like Richard Feynman and Norio Taniguchi began to explore the fascinating possibilities of manipulating matter at an atomic and molecular level. In the 1950s, the field gained momentum with the development of the scanning tunneling microscope (STM) by Gerd Binnig and Heinrich Rohrer, which allowed for the visualization of individual atoms. The 1980s marked a significant leap forward with the invention of the atomic force microscope (AFM) by Binnig, Rohrer, and Calvin Quate, further enhancing the ability to manipulate and understand nanoscale structures. This period saw the birth of the term "nanotechnology" itself, coined by Eric Drexler in 1986. The 1990s witnessed a surge in nanotechnology research, with the establishment of national nanotechnology initiatives around the world, including the U.S. National Nanotechnology Initiative in 2000. This era has been characterized by rapid advancements, leading to the development of various nanomaterials and nanodevices with promising applications in medicine. Despite the remarkable progress, the field continues to face challenges in scaling up research findings into practical and safe medical applications, underscoring the ongoing evolution of nanotechnology."
    }
]
```
'''

import utils
import re
import json
def convert_string_to_json_value(input_string):
    # 将字符串直接转换为 JSON 格式的值
    json_string = json.dumps({"Content": input_string}, ensure_ascii=False)
    return json_string
# tils = utils.extract_json_from_text(input)

c_pattern = r'"Content":\s*"([^}]*)"'
id_pattern = r'"ParagraphID":\s*"([^ ]*)"'

matches = re.findall(c_pattern, input)
for match in matches:
    print(match)
    output = convert_string_to_json_value(match)
    print(output)


import json




