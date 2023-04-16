#!/usr/bin/env python3
import io
import json
import os
import pickle
import re
import subprocess
import time

import openai
import pdfplumber
import tiktoken
import urllib3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import ddg
from selenium import webdriver
from googlesearch import search

driver = webdriver.Chrome()

load_dotenv()

# Engine configuration

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
        os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    from extensions.argparseext import parse_arguments

    OBJECTIVE, INITIAL_TASK, OPENAI_API_MODEL, DOTENV_EXTENSIONS = parse_arguments()

# Load additional environment variables for enabled extensions
if DOTENV_EXTENSIONS:
    from extensions.dotenvext import load_dotenv_extensions

    load_dotenv_extensions(DOTENV_EXTENSIONS)

openai.api_key = OPENAI_API_KEY

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
MAX_TOPICS = int(os.getenv("MAX_TOPICS"))
MAX_QUERIES = int(os.getenv("MAX_QUERIES"))
MAX_BROWSE = int(os.getenv("MAX_BROWSE"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS"))


url_cache = {}
if os.path.exists('url_cache'):
    with open('url_cache', 'rb') as f:
        url_cache = pickle.load(f)
search_cache = {}
if os.path.exists('search_cache'):
    with open('search_cache', 'rb') as f:
        search_cache = pickle.load(f)
openai_cache = {}
if os.path.exists('openai_cache'):
    with open('openai_cache', 'rb') as f:
        openai_cache = pickle.load(f)

def save_summaries(summaries):
    with open('summaries', 'wb') as f:
        pickle.dump(summaries, f)


def get_url(url):
    if url in url_cache:
        return url_cache[url]
    for a in range(3):
        try:
            driver.get(url)
            # Get the page source
            time.sleep(10)
            page_source = driver.page_source
            # Parse the HTML with Beautiful Soup
            soup = BeautifulSoup(page_source, 'html.parser')
            # Extract the desired content using Beautiful Soup's methods
            # Get the concatenated text of all elements, with optional separator and strip options
            url_cache[url] = soup
            with open('url_cache', 'wb') as f:
                pickle.dump(url_cache, f)
            return soup
        except Exception as ex:
            print(f"get_url '{url}'")
            print('get_url', ex)
            time.sleep(30)
            continue
    return ''

def read_pdf(url: str) -> str:
    http = urllib3.PoolManager()
    temp = io.BytesIO()
    temp.write(http.request("GET", url).data)
    with pdfplumber.open(temp) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text

def scrape_text(url):
    if url.lower().endswith(".pdf"):
        print(f'WILL DOWNLOAD AND PARSE PDF: {url}')
        return read_pdf(url)

    soup = get_url(url)
    if soup == '':
        return ''
    text = soup.get_text(separator=' ', strip=True)
    return text


def extract_hyperlinks(soup):
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append((link.text, link['href']))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    soup = get_url(url)

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)


def split_text(text):
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        while count_tokens(paragraph) > MAX_TOKENS:
            paragraph = paragraph[:len(paragraph) - 100]
        if current_length + count_tokens(paragraph) + 1 <= MAX_TOKENS:
            current_chunk.append(paragraph)
            current_length += count_tokens(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def create_message(chunk, question):
    # return {
    ##    "role": "user",
    #    "content": f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text."
    # }
    return f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text."


def summarize_text(text, question):
    if not text:
        return "Error: No text to summarize"

    text_length = len(text)
    # print(f"Text length: {text_length} characters")

    if text_length < 1000:
        return text

    summaries = []
    chunks = list(split_text(text))

    for i, chunk in enumerate(chunks):
        # print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        messages = create_message(chunk, question)
        # print(f'chunk {i} messages {messages}')

        summary = openai_call(messages, max_tokens=300)
        # print(f'chunk {i} summary {summary}')
        summaries.append(summary)

    print(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    messages = create_message(combined_summary, question)
    # print('combined messages', messages)

    final_summary = openai_call(messages, max_tokens=300)
    # print('final_summary', final_summary)

    return final_summary

def google_search(query):
    if query in search_cache:
        return search_cache[query]
    search_results = []

    #results = ddg(query)
    results = search(query, advanced=True)
    if results is None:
        return []

    for j in results:
        search_results.append({'Url': j.url, 'Title': j.title})

    if len(search_results) == 0:
        return []

    search_cache[query] = json.dumps(search_results, ensure_ascii=False, indent=4)
    with open('search_cache', 'wb') as f:
        pickle.dump(search_cache, f)
    return search_cache[query]

def google_search_old(query, num_results=8):
    if query in search_cache:
        return search_cache[query]
    print("google_search", query)
    search_results = []
    for j in ddg(query, max_results=num_results):
        search_results.append(j)

    if len(search_results) == 0:
        return []

    search_cache[query] = json.dumps(search_results, ensure_ascii=False, indent=4)
    with open('search_cache', 'wb') as f:
        pickle.dump(search_cache, f)
    return search_cache[query]


def browse_website(url, question):
    print("browse_website", url)
    summary = get_text_summary(url, question)
    links = get_hyperlinks(url)

    # Limit links to 10
    if len(links) > 10:
        links = links[:10]

    result = f"""Website Content Summary: {summary}\n\nLinks: {links}"""

    return result


def get_text_summary(url, question):
    text = scrape_text(url)
    summary = summarize_text(text, question)
    return """ "Result" : """ + summary


def get_hyperlinks(url):
    link_list = scrape_links(url)
    return link_list



def add_openai_cache(prompt, response):
    # print(f'Prompt: {prompt}\n\nResponse: {response}\n\n\n')
    openai_cache[prompt] = response
    with open('openai_cache', 'wb') as f:
        pickle.dump(openai_cache, f)
    return response


def openai_call(
        prompt: str,
        model: str = OPENAI_API_MODEL,
        temperature: float = 0.5,
        max_tokens: int = 100,
):
    if prompt in openai_cache:
        return openai_cache[prompt]

    while True:
        try:
            if model.startswith("llama"):
                # Spawn a subprocess to run llama.cpp
                cmd = cmd = ["llama/main", "-p", prompt]
                result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                return add_openai_cache(prompt, result.stdout.strip())
            elif not model.startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return add_openai_cache(prompt, response.choices[0].text.strip())
            else:
                # Use chat completion API
                messages = [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return add_openai_cache(prompt, response.choices[0].message.content.strip())
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break





def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    return len(encoding.encode(text))


def insert_prompt(prompt, params):
    length_per_value = 5000
    while True:
        ret = prompt
        for key, value in params.items():
            if type(value) == str:
                ret = ret.replace('{' + key + '}', value[:length_per_value])
            elif type(value) == int:
                ret = ret.replace('{' + key + '}', str(value)[:length_per_value])
            elif type(value) == list:
                ret = ret.replace('{' + key + '}', ('\n'.join(value))[:length_per_value])
            elif type(value) == dict:
                ret = ret.replace('{' + key + '}', str(value['total'] - value['used'])[:length_per_value])
            else:
                raise Exception('insert_prompt: incorrect type of parameter ' + str(type(value)) + ', key=' + key)
        tokens = count_tokens(ret)
        if tokens < MAX_TOKENS:
            break
        else:
            print('WARNING! insert_prompt truncating text', prompt, params)
            length_per_value = length_per_value - 500
            if length_per_value <= 0:
                raise Exception('insert_prompt: length_per_value<0 for prompt ' + prompt)
    #if ret.find('{') != -1 or ret.find('}') != -1:
    #    print('WARNING! insert_prompt possible has unfilled variables', {'prompt': prompt, 'params': params, 'ret': ret})
    return ret


search_prompt = '''As a Search Agent, your task is to generate a list of queries related to the topic of {topic} 
for further investigation. Please create list of relevant
queries for a web search engine, such as Google, that will help gather information on the specific topic.
Please note that final task is {objective}.
Critical: reply with list of search queries (no more than {max_queries}), each on a different line'''


def get_search_queries(topic, objective):
    try:
        prompt = insert_prompt(search_prompt, {'topic': topic, 'objective': objective, 'max_queries': MAX_QUERIES})
        queries = openai_call(prompt).split('\n')
        queries = [x.strip() for x in queries]
        return [x for x in queries if x != '']
    except Exception as ex:
        print(f'In get_search_queries ERROR={ex}')
        return []


decompose_prompt = '''As a Decomposition Agent, your task is to create a list of topics to investigate for the main task of 
{objective}. Consider inputs from other agents: previous collected summaries {topic_summaries}. 
Critics from the previous attempts: {critics}.
Generate a comprehensive list of relevant topics that need to be explored further.
Important: respond with only list of topics (max {max_topics}), each in separate line'''


def split_tasks(text):
    ret = [x.strip() for x in text.split('------')]
    ret = [x for x in ret if x != '']
    return ret


def decompose(objective, topic_summaries, critics, prev_topics):
    try:
        if critics != 'N/A':
            critics += f'\nPlease make sure to cover the new topics mentioned in critics. The topics {prev_topics} are already explored.'
        prompt = insert_prompt(decompose_prompt,
                               {'objective': objective, 'topic_summaries': topic_summaries, 'critics': critics, 'max_topics': MAX_TOPICS})
        topics = openai_call(prompt).split('\n')
        topics = [x.strip() for x in topics]
        return [x for x in topics if x != '']
    except Exception as ex:
        print(f'In decompose ERROR={ex}')
        return []


filtering_prompt = '''Search results: {search_results}

As a Search Result Filtering Agent, your task is to select {max_browse} links from the list below,
which are most relevant to the topic {topic}, needed for the task {objective}.
Note: you do not need to include anything else than URLs in your response, as they will be retrieved later. 
You need to return ONLY URLs!
Important: to enable automatic parsing, your response must be formatted as
#. http://first url
#. https://second url
'''


def get_filtered_results(objective, topic, search_results):
    pattern = r'^\d+\.'
    try:
        prompt = insert_prompt(filtering_prompt,
                               {'objective': objective, 'topic': topic, 'search_results': search_results,
                                'max_browse': MAX_BROWSE})
        urls = openai_call(prompt).split('\n')
        urls = [re.sub(pattern, '', x.strip()).strip() for x in urls]
        return [x for x in urls if x != '' and x.startswith('http') and search_results.find(x) != -1]
    except Exception as ex:
        print(f'In get_filtered_results ERROR={ex}')
        return []


browse_prompt = '''As a Browse Agent, your task is to view websites and retrieve information related to the specific
 topic of {topic} needed for the objective {objective}. Process the text below which is part of a website content retrieved
 with a query {query}, and provide a summary of the findings, including key insights, relevant details, and any noteworthy information related to 
 the topic. Important: please answer with a single `-` sign if the text is not relevant to the topic. Text:
 
 {search_results}'''


def browse_and_summarize(url, query, topic, objective):
    try:
        text = scrape_text(url)
        chunks = list(split_text(text))
        all = []

        for chunk in chunks:
            prompt = insert_prompt(browse_prompt,
                                   {'objective': objective, 'topic': topic, 'query': query, 'search_results': chunk})
            summary = openai_call(prompt).strip()
            if summary != '-':
                all.append(summary)

        if len(all) == 0:
            return ''
        return f'Summary for url {url}: ' + '\n'.join(all)
    except Exception as ex:
        print(f'In browse_and_summarize ERROR={ex}')
        return ''


summarize_prompt = '''Responses from the Browse agents: {all_results}

As a Summarization Agent, your task is to analyze and summarize the responses provided by the 
Browse Agents for topic {topic}. Based on the information retrieved, generate a final answer by consolidating 
the findings, highlighting important details, and providing a very accurate concise summary of the results. Note than final objective
is {objective}. Important: if information is missing in the results above, answer that it is missing. Do not try to remember or make up the answer.

Please format your output as following:
Thought: <your thoughts>
Reasoning: <your reasoning>
Answer: <final answer, may be up 2000 words, it will be passed to the higher level agent>'''


def summarize_topic(summaries, topic, objective):
    try:
        prompt = insert_prompt(summarize_prompt,
                               {'objective': objective, 'topic': topic, 'all_results': '\n'.join(summaries)})
        summary = openai_call(prompt, max_tokens=2000)
        print(f'RAW SUMMARY={summary}')
        if summary.find('Answer:') != -1:
            summary = summary[summary.find('Answer:') + len('Answer:'):].strip()
        return f'Summary for topic {topic}:\n' + summary
    except Exception as ex:
        print(f'In summarize_topic ERROR={ex}')
        return ''


final_prompt = '''Summaries from the agents: {all_results}

As a Final Summarization Agent, your task is to analyze and summarize the responses provided by the 
previous Agents for different topics. Based on the information retrieved, generate a final answer by consolidating 
the findings, highlighting important details, and providing a concise summary of the results. Note than final objective
is {objective}. Important: if information is missing in the results above, answer that it is missing. Do not try to remember or make up the answer.
Please format your output as following:
Thought: <your thoughts>
Reasoning: <your reasoning>
Answer: <final answer, may be up 1000 words, it will be passed to human>'''


def create_final_answer(summaries, objective):
    prompt = insert_prompt(final_prompt,
                           {'objective': objective, 'all_results': '\n'.join(summaries)})
    summary = openai_call(prompt, max_tokens=2000)
    print(f'RAW SUMMARY={summary}')
    if summary.find('Answer:') != -1:
        summary = summary[summary.find('Answer:') + len('Answer:'):].strip()
    return summary


critic_task_prompt = '''As a Critic Agent, your task is to evaluate the success of the task completion based on the final 
answer provided by the Summarization Agent. Assess the accuracy, comprehensiveness, and relevance of the information presented. 
Provide feedback and suggestions for improvement, and identify if further investigation or topic exploration is needed.
Critical: please format your response as following: in the first line only one word: yes/no 
(yes - if the agent has fully complete the objective and you see nothing to improve there)
In the next lines: up to 200-word explanation what is wrong and what should be changed in the next attempt.
Objective: {objective}
Agent's final answer: {final_answer}'''


def critic_agent(final_answer, objective):
    text = openai_call(insert_prompt(critic_task_prompt,
                                     {'objective': objective, 'final_answer': final_answer}), max_tokens=200)
    return text


def task_complete(critic):
    s = critic.strip()
    if '\n' in s:
        first_word = s[:s.index('\n')].strip().lower()
    else:
        first_word = s
    if first_word.find("yes") != -1:
        return True
    elif first_word.find("no") != -1:
        return False
    else:
        raise Exception(f"task_complete: cannot parse response '{critic}'")


critics = 'N/A'
topic_summaries = []
summaries = {}
if os.path.exists('summaries'):
    with open('summaries', 'rb') as f:
        summaries = pickle.load(f)

final_answer = ''

for attempt in range(MAX_ATTEMPTS):
    print(f'attempt={attempt}')

    topics = decompose(OBJECTIVE, topic_summaries, critics, prev_topics=', '.join(summaries.keys()))
    for topic in topics[:MAX_TOPICS]:
        print(f'TOPIC={topic}')
        search_queries = get_search_queries(topic, OBJECTIVE)
        if topic not in summaries:
            summaries[topic] = {}

        for query in search_queries[:MAX_QUERIES]:
            print(f'SEARCH_QUERY={query}')
            search_results = google_search(query)
            if len(search_results) == 0:
                print('ERROR: NO RESULT FOR', query)
            else:
                filtered_results = get_filtered_results(OBJECTIVE, topic, search_results)
                print(f'FILTERED URLS={filtered_results}')
                for url in filtered_results[:MAX_BROWSE]:
                    if url not in summaries[topic]:
                        summaries[topic][url] = browse_and_summarize(url, query, topic, OBJECTIVE)
                        save_summaries(summaries)

        print(f'SUMMARIES={summaries[topic]}')
        if len(list(summaries[topic].values())) == 0:
            print('ERROR: NO SUMMARIES')
        else:
            topic_summary = summarize_topic(list(summaries[topic].values()), topic, OBJECTIVE)
            topic_summaries.append(topic_summary)

    print(f'TOPIC_SUMMARIES={topic_summaries}')
    if len(topic_summaries) == 0:
        print('ERROR: NO TOPIC SUMMARIES')
        break
    else:
        final_answer = create_final_answer(topic_summaries, OBJECTIVE)
        print(f'FINAL_ANSWER={final_answer}')
        critics = critic_agent(final_answer, OBJECTIVE)
        print(f'CRITICS={critics}')
        if task_complete(critics):
            break

save_summaries(summaries)

print(f'FINAL_ANSWER={final_answer}')
