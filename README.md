# Sentiment Analysis



This project is part of the IFT 6135B - H2024 course under Prof. Aaron Courville. It contains the implementation and report for the sentiment analysis project on the Yelp Polarity dataset using Recurrent Neural Networks (RNNs) and Transformer models.  The goal of this assignment is to compare the performance of RNNs with various configurations of the Transformer model for sentiment analysis. The Yelp Polarity dataset is pre-processed using a BERT-based Hugging Face tokenizer, which outputs the padded sequence with a corresponding mask denoting where the padding is, and sentiment label.

**Problem 1: Implementing an LSTM Encoder-Decoder with Soft Attention**

- Implemented a custom LSTM class using PyTorch without relying on the built-in `nn.LSTM` or `nn.LSTMCell` modules.
- Created a bidirectional LSTM encoder with dropout on the embedding layer.
- Developed a decoder with an option to include a self-attention mechanism.

**Problem 2: Implementing a Transformer**

- Implemented Layer Normalization (LayerNorm) without using PyTorch's built-in `nn.LayerNorm` module.
- Created a multi-head scaled dot-product attention mechanism with padded masking.
- Developed a miniature Transformer model with self-attention and a feed-forward neural network, with skip-connections.


**Configurations**

1. LSTM, no dropout, encoder only
2. LSTM, dropout, encoder only
3. LSTM, dropout, encoder-decoder, no attention
4. LSTM, dropout, encoder-decoder, attention
5. Transformer, 2 layers, pre-normalization
6. Transformer, 4 layers, pre-normalization 
7. Transformer, 2 layers, post-normalization
8. Fine-tuning BERT model
