This model has been trained without supervision following the approach described in [Towards Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118). The associated GitHub repository is available here https://github.com/facebookresearch/contriever.

## Usage (HuggingFace Transformers)
Using the model directly available in HuggingFace transformers requires to add a mean pooling operation to obtain a sentence embedding.

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model = AutoModel.from_pretrained('facebook/contriever')

sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

# Apply tokenizer
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
outputs = model(**inputs)

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
```