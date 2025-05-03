You can create a simple slide show here by adding files to the _posts directory.

The website for this repository is viewable at: https://tulane-cmps6730.github.io/sp2-25-automotive . You will have a different url for your team's repository -- e.g., if your team's repository is `project-alpha`, it will be at https://tulane-cmps6730.github.io/project-alpha.

Here is a tutorial on the Markdown syntax: https://guides.github.com/features/mastering-markdown/

I added the reveal.js submodule by `git submodule add https://github.com/hakimel/reveal.js reveal.js`

# Automotive Review Summarizer
Andrew Selius
CMPS6730, Spring 2025

## Problem Statement and Solution Objective
The automotive industry can be daunting to navigate for inexperienced shoppers. The variety of makes, models, and trim levels can present shoppers with an overwhelming amount of information. Even when knowledgeable in this information, shoppers may still have lingering questions on the quality of automobile between brands, as some are considered more reputable than others. The aim of this experiment is to create an interactive model, powered by Retrieval-Augmented Generation (RAG), to quickly deliver prospective buyers a summary of brand features and reliability, using both general brand sentiment and reviews of specific models of vehicles.

## Approach

The data used for this experiment consists of general brand overviews, general brand reviews, summary rankings and their explanations, and a variety of reviews from reputable automotive journalist publications. Sources include US News, Motor Trend, Consumer Reports, J.D. Power, Kelley Blue Book, and Edmunds. Data was collected via the respective source’s webpages and organized into an input CSV file, with labels for each associated brand given an entry. 100 entries from various sources were used for model development, each corresponding to one of ten brands: Honda, Toyota, Subaru, Jeep, B.M.W., Mazda, Tesla, Dodge, Ford, and Chevrolet.

The language model used is Google’s **T5-Base**, with the **all-MiniLM-L6-V2** embedder. The embedder was used in combination with FAISS to generate embeddings of the preprocessed data. Evaluation was conducted by comparing similar models and embedders to determine the best combination. Evaluation metrics used were: ROUGE-1, ROUGE-2, ROUGE-2, BLEU, and subjective analysis of response grammar and cohesiveness.

## Results

After testing model-embedder combinations against a sample set of queries and reference responses, the following results were obtained: <br/><br/>
![Model-Embedder Rouge-2 Scores](/docs/MERouge2.png)
<br/><br/>
![Model-Embedder Rouge-L Scores](/docs/MERougeL.png)
<br/><br/>
![Model-Embedder Bleu Scores](/docs/MEBleu.png)
<br/><br/>
![Bar Plot of Rouge Scores by Model](/docs/MRougeBar.png)
<br/><br/>
![Bar Plot of Rouge Scores by Embedder](/docs/ERougeBar.png)
<br/><br/>
Below are some sample demo outputs from the Flask webpage:
<br/><br/>
![Demo output: "Is Honda reliable?"](/docs/Demo1.png)
<br/><br/>
![Demo output: "Are BMWs sporty?"](/docs/Demo2.png)
<br/><br/>
![Demo output: "Are Chevrolets practical cars?"](/docs/Demo3.png)
<br/>

## Conclusions

The results of this project have shown promise in addressing the need for a tool to answer consumer questions about different cars. While there is much room for improvement on answer quality, the model has shown its ability to deliver a relevant response to the user's ask. Future iterations of this project may aim to fine-tune answer quality to more precisely address a user's ask.


