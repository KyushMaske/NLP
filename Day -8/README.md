Explored  more about CausalLM.

**Causal Language Models** (CausalLMs) generate text in a sequential, word-by-word manner, where each word is predicted based on the preceding context. This type of model is called "causal" because each word in the sequence is causally dependent on the previous words, meaning that the prediction of the next word is influenced only by the words that came before it.

**Key Characteristics:**
Autoregressive Nature: These models generate text in a left-to-right fashion, using their previous outputs as part of the input for predicting the next word.
Unidirectional Context: Predictions are made considering only the past context, not the future context.
Loss Function:
The loss function for CausalLMs is designed to train the model to predict the next word in the sequence accurately. The most commonly used loss function is the cross-entropy loss, which measures the difference between the predicted probability distribution of the next word and the actual word in the training data.

[CasualLM](https://towardsdatascience.com/training-causallm-models-part-1-what-actually-is-causallm-6c3efb2490ec)