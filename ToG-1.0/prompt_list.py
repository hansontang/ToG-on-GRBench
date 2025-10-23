# prompt_list.py

# 用于 link_qa.py，从问题中提取核心实体名称
ENTITY_LINKING_PROMPT = """
Your task is to accurately extract the core topic entity name from the user's question.
You need to understand the user's intent and return only the entity name itself, without extra words like "brand of" or "item".
Please strictly return the result in the JSON format {{"entity_name": "..."}}. If no clear entity is present in the question, return {{"entity_name": null}}.

---
[Example 1]
Question: "Could you specify the brand of Blackberry Playbook 7-Inch Tablet (64GB)?"
Answer: {{"entity_name": "Blackberry Playbook 7-Inch Tablet (64GB)"}}

---
[Example 2]
Question: "What brand does the item Sassy Developmental Bath Toy, Catch and Count Net belong to?"
Answer: {{"entity_name": "Sassy Developmental Bath Toy, Catch and Count Net"}}

---
[New Question]
Question: "{question}"
Answer:
"""

# 用于 grbench_func.py，让 LLM 对候选关系进行打分
RELATION_PRUNE_PROMPT = """
You are an expert in evaluating the relevance of knowledge graph relations to a given question.
Based on the user's question and the topic entity, score the following candidate relations on a scale from 0.0 to 1.0, where 1.0 is most relevant.
Return the scores in a JSON list format. The order of scores must match the order of the relations in the input.

---
[Example]
Question: "What is the price of the item 'The Sherlock Holmes Audio Collection' and what brand is it?"
Topic Entity: "The Sherlock Holmes Audio Collection"
Relations: ["brand", "also_viewed_item", "bought_together_item"]
Answer: [1.0, 0.1, 0.3]

---
[New Task]
Question: "{question}"
Topic Entity: "{topic_entity}"
Relations: {relations}
Answer:
"""

# 用于 grbench_func.py，让 LLM 对候选实体进行打分
ENTITY_PRUNE_PROMPT = """
You are an expert in evaluating the relevance of knowledge graph entities to a given question, considering the relation path taken to reach them.
Based on the user's question and the current reasoning path, score the following candidate entities on a scale from 0.0 to 1.0.
Return the scores in a JSON list format. The order of scores must match the order of the entities in the input.

---
[Example]
Question: "What brand is the book 'The Lord of the Rings'?"
Current Path: "'The Lord of the Rings' -> has_brand -> ?"
Entities: ["J.R.R. Tolkien", "Houghton Mifflin Harcourt", "George Allen & Unwin", "Winged Helmet"]
Answer: [0.1, 0.9, 0.9, 0.0]

---
[New Task]
Question: "{question}"
Current Path: "{current_path}"
Entities: {entities}
Answer:
"""

# 用于 grbench_func.py，判断当前信息是否足够回答问题
REASONING_PROMPT = """
You are a reasoning agent. Your task is to determine if the collected knowledge graph paths provide sufficient information to definitively answer the user's question.
Respond with a JSON object: {{"sufficient": true/false}}.

---
[Example 1]
Question: "What is the brand of 'The Sherlock Holmes Audio Collection'?"
Knowledge Paths:
- "'The Sherlock Holmes Audio Collection' -> has_brand -> 'BBC Audiobooks'"
Answer: {{"sufficient": true}}

---
[Example 2]
Question: "What is the price of 'The Sherlock Holmes Audio Collection'?"
Knowledge Paths:
- "'The Sherlock Holmes Audio Collection' -> has_brand -> 'BBC Audiobooks'"
- "'The Sherlock Holmes Audio Collection' -> also_bought_item -> 'The Complete Sherlock Holmes'"
Answer: {{"sufficient": false}}

---
[New Task]
Question: "{question}"
Knowledge Paths:
{knowledge_paths}
Answer:
"""

# 用于 grbench_func.py，生成最终答案
ANSWER_GENERATION_PROMPT = """
You are a machine that answers questions based on provided facts.
Your task is to answer the user's question using ONLY the provided knowledge paths.
Do NOT add any extra words, explanations, or introductory phrases like "The answer is" or "Based on the information".
Respond ONLY with the answer entity/value itself.

---
[Context]
Question: "{question}"
Knowledge Paths:
{knowledge_paths}

---
[Answer]
"""
