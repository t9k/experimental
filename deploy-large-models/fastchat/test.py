import openai
if __name__ == "__main__":
    openai.api_base = "http://100.64.4.62:31185/v1"
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
        model="chatglm2-6b",
        messages=[
            {"role": "user", "content": "清华大学和北京大学哪个好"}
        ],
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)
