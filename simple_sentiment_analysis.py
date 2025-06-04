import json
from transformers import pipeline
import matplotlib.pyplot as plt

# 创建情绪分析pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 读取帖子数据
with open("sample_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

# 结果记录
results = {"POSITIVE": 0, "NEGATIVE": 0}

# 情感识别函数
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label']

# 对每条内容进行分析
for post in posts:
    sentiment = analyze_sentiment(post["content"])
    print(f"User: {post['user']} | Sentiment: {sentiment}")
    results[sentiment] += 1

# 可视化结果（调整键名为大写）
plt.bar(
    [k.capitalize() for k in results.keys()],
    results.values(),
    color=["green", "red"]
)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.show()
