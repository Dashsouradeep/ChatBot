from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Define a context (information about mining in India)
context = """
Mining in India is regulated by various laws, acts, and rules. The Mines and Minerals (Development and Regulation) Act, 1957, is one of the key legislations governing mining activities. 
This act outlines the framework for granting mining leases, royalty rates, and mineral conservation and development rules. Additionally, the Environmental Impact Assessment (EIA) process and the Forest Conservation Act are crucial for ensuring compliance with environmental regulations in mining operations.
"""

# Define a user query
question = "What is the Mines and Minerals (Development and Regulation) Act, 1957?"

# Tokenize the input
inputs = tokenizer(question, context, return_tensors='pt')

# Perform question answering
start_scores, end_scores = model(**inputs)

# Get the answer span
start_idx = torch.argmax(start_scores)
end_idx = torch.argmax(end_scores) + 1  # Add 1 to include the end token
answer_span = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))

# Print the answer
print("Answer:", answer_span)
