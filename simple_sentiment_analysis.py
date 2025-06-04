import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# 加载情绪分类模型
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 读取帖子数据
with open("sample_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

# 结果记录
results = {"Positive": 0, "Negative": 0}

# 情感识别函数
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

# 对每条内容进行分析
for post in posts:
    sentiment = analyze_sentiment(post["content"])
    print(f"User: {post['user']} | Sentiment: {sentiment}")
    results[sentiment] += 1

# 可视化结果
plt.bar(results.keys(), results.values(), color=["green", "red"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.show()
