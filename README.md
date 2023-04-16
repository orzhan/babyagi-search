<h1 align="center">
 babyagi-search

</h1>

# Objective
A web search, browsing, and summarization system to get better search results from multiple web sites. It is based on [babyagi](https://github.com/yoheinakajima/babyagi) and also uses some ideas from [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT)
Compared to systems like Bing, it is configurable (level of task decomposition can be specified).

# How It Works<a name="how-it-works"></a>
First, task is decomposed into topics. For each topic, several search queries are generated. Search results for each query are filtered. Selected websites are browsed, and a summary is created for the selected topic. Results from all websites for a topic are summarized again. And in the end final summary from all topics is written. A critic verifies if the task is done, and if not, a second pass of searching is done based on the results of the first pass.

# How to Use<a name="how-to-use"></a>
To use the script, you will need to follow these steps:

1. Clone the repository via `git clone https://github.com/orzhan/babyagi-search.git` and `cd` into the cloned repository.
2. Install the required packages: `pip install -r requirements.txt`
3. Copy the .env.example file to .env: `cp .env.example .env`. This is where you will set the following variables.
4. Set your OpenAI key in the OPENAI_API_KEY, OPENAPI_API_MODEL variables.
5. Set the objective (what result do you want to get) in the OBJECTIVE variable.
6. Run the script.

All optional values above can also be specified on the command line.

# Supported Models<a name="supported-models"></a>

This script works with all OpenAI models, as well as Llama through Llama.cpp. Default model is **gpt-3.5-turbo**. To use a different model, specify it through OPENAI_API_MODEL or use the command line.

## Llama

Download the latest version of [Llama.cpp](https://github.com/ggerganov/llama.cpp) and follow instructions to make it. You will also need the Llama model weights.

 - **Under no circumstances share IPFS, magnet links, or any other links to model downloads anywhere in this repository, including in issues, discussions or pull requests. They will be immediately deleted.**

After that link `llama/main` to llama.cpp/main and `models` to the folder where you have the Llama model weights. Then run the script with `OPENAI_API_MODEL=llama` or `-l` argument.
