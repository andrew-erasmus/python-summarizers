import requests
from transformers import BartForConditionalGeneration, BartTokenizer

# Function that will take given text and summarize it with a maximum length that the user decides
def summarize_article(article_text, max_length, min_length):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    paragraph_text=input("Enter a paragraph you'd like to summarize: ") # paragraph to be summarized
    min_length=eval(input("Minimum length?: ")) # min word count to be used
    max_length=eval(input("Maximum length?: ")) # max word count to be used

    print('Loading . . .')

    summary = summarize_article(paragraph_text, max_length, min_length)
    print("\nSummarized Article:")
    print(summary)