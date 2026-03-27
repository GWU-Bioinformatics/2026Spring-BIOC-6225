This class will build a retrieval augmentation system piece by piece. We start with a framework, then add the core components of a modern RAG. We will compare the scripts with a diff to see which sections have been swapped.

**Script 1** comes from [Bill Chambers](https://towardsdatascience.com/a-beginners-guide-to-building-a-retrieval-augmented-generation-rag-application-from-scratch-e52921953a5d/). While not a realistic modern RAG, it has a strong foundation that we can build on. We'll examine the pieces of this code, what they do, and what needs to be done in order to make it a functional, modern RAG.
- This script does have a corpus, a retrieval step, a prompt that includes retrieved content, and a generation step. However, it uses simple lexical overlap, and does not use a strong retriever.

**Script 2** will build on script 1 by adding basic tokenization, tf-idf vectors, cosine similarity, and top_k selection.
- We introduce basic regex-based tokenization (with stopword removal). We also include a score threshold so that we don't waste time with poor quality hits.

**Script 3** will begin to make use of scikit-learn. Now that some of the foundational concepts are in place, we use a high quality library to handle them.
- Using scikit-learn is common in many modern approaches, however, this script is essentially a lateral move in terms of approach. There are no conceptual changes, only changes to the tools that are used. It keeps the same retrieval logic as the last script, but it replaces the TF-IDF implementation with scikit-learn.

**Script 4** will swap lexical retrieval for embedding-based semantic retrieval.
- Instead of weighted term vectors used in the last two scripts, this version will embed a query, embed the documents, and then compare them in embedding space. 

Once you've downloaded llama3, please be sure to run the following steps to be ready to run the scripts:
1. `ollama pull llama3`
2. `ollama pull embeddinggemma`
3. `ollama serve`

If you do not have sklearn installed, please also install that:
`pip install scikit-learn`

Each of the scripts can be run with `python3 <scriptname.py>`
