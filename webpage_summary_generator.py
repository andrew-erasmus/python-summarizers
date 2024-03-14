from transformers import BartForConditionalGeneration, BartTokenizer
import requests
from bs4 import BeautifulSoup

# Function that will take scraped article and summarize it with a maximum length of 250 words
def summarize_article(article_text, max_length=250):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

## Function to scrape content of the aricle 
def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        all_text = soup.get_text(separator='\n', strip=True)
        return all_text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    webpage_url = input("Enter a url of a site you'd like to summarise: ") # URL for site to be scraped
    print('Loading . . .')
    webpage_text = scrape_webpage(webpage_url)

    if webpage_text:
        summary = summarize_article(webpage_text)
        print("\nSummarized Article:")
        print(summary)
    else:
        print("Webpage scraping failed.")