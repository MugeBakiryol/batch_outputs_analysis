# Batch Outputs Analysis: Probing Deep Learning Models with SentEval

## Overview
This repository contains resources from my internship project at Tilburg University, where I investigated the linguistic properties encoded in sentence embeddings through probing tasks. The project focused on replicating the methodology outlined in *What You Can Cram into a Single Vector: Probing Sentence Embeddings for Linguistic Properties* by Conneau et al. (2018), while extending its applicability to modern encoder architectures.

## Key Features
- **Replication and Extension:** Replicated and built upon the probing task framework from Conneau et al. (2018) to evaluate linguistic properties such as syntax, semantics, and surface-level features.
- **Pipeline Development:** Designed a robust and modular pipeline using PyTorch, NumPy, and SentEval for evaluating sentence embeddings.
- **Probing Tasks:** A variety of linguistic evaluation tasks, including:
  - Sentence Length
  - Tree Depth
  - Tense Detection
  - Subject/Object Number
  - Coordination Inversion
  - Odd-Man-Out
- **Encoder Architectures Evaluated:**
  - **BiLSTM-last:** Captures cumulative contextual information using the final hidden state.
  - **BiLSTM-max:** Highlights key features using maximum pooling across hidden states.
  - **Gated CNN:** Excels at detecting hierarchical linguistic patterns through convolutional and gating mechanisms.

## Contents
- `data/`: Directory contains probing task datasets from SentEval, which are used to evaluate linguistic properties in sentence embeddings.
- `examples/`: Python script examples for running the experiments.

Required Data

To replicate the tasks, download the necessary datasets from the SentEval repository:

https://github.com/facebookresearch/SentEval/tree/main/data

Ensure the datasets are placed in the appropriate directory before running the experiments.

## How to Use
1. **Clone the repository:**
   ```bash
   git clone https://github.com/facebookresearch/SentEval/tree/main/data
   cd SentEval-main
   ```
2. **Install dependencies:**
   ```bash
   python setup.py
   ```
3. **Run the pipeline:**
   ```bash
   python senteval_script_v3.py
   ```
4. **View results:**

## Research Background
The primary goal of this internship was to evaluate linguistic representations using sentence embeddings and probing tasks. This project replicated Conneau et al. (2018)'s work and extended it by analyzing multiple encoder architectures, including BiLSTM and Gated CNN. Key findings include:

- **BiLSTM-max:** Superior performance in semantic tasks due to maximum pooling, emphasizing key features in representation.
- **Gated CNN:** Demonstrated strong results in structural tasks, capturing both local and global dependencies effectively.

These insights highlight encoder-specific strengths and weaknesses, advancing the interpretability of AI models and their practical applications in tasks like translation, sentiment analysis, and question answering.

## Future Directions
- Integrate transformer-based models like BERT and GPT into the pipeline.
- Develop unsupervised probing tasks to uncover deeper linguistic insights.
- Extend the framework to support multimodal tasks by incorporating text, images, and audio.

## Acknowledgments
This project was conducted under the supervision of **Dr. Noortje Venhuizen** at Tilburg University, School of Humanities and Digital Sciences. I am especially grateful for her guidance and mentorship throughout the project. I also extend my sincere thanks to **Dr. Harm Brouwer** and **Rémy Marro** for their valuable feedback and discussions, which enriched the depth and rigor of this research.

## References
1. Conneau, A., Kruszewski, G., Lample, G., Barrault, L., & Baroni, M. (2018). *What You Can Cram into a Single Vector: Probing Sentence Embeddings for Linguistic Properties*. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics.
2. Belinkov, Y. (2022). *Probing classifiers: Promises, shortcomings, and advances*. Computational Linguistics.
3. Hewitt, J., & Liang, P. (2019). *Designing and interpreting probes with control tasks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.
4. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient estimation of word representations in vector space*. arXiv preprint, arXiv:1301.3781.
5. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global vectors for word representation*. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing.
