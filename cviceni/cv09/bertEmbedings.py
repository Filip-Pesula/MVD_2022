from transformers import  BertModel,BertTokenizer
from torch.nn import  CosineSimilarity


import torch

class Embedings:
    def __init__(self) -> None:
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dist = CosineSimilarity(dim=0)

    def getWordWec(self,text,verbouse = 1):
        tokenized_text = self.tokenizer.tokenize(text)
        print(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        hidden_states = None
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # becase we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        token_embeddings.size()

        token_vecs_cat = []
        for token in token_embeddings:
            
            # `token` is a [12 x 768] tensor

            # Concatenate the vectors (that is, append them together) from the last 
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)
        if verbouse > 0:
            print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
        return tokenized_text,token_vecs_cat

    def processSentences(self,sentences,verbouse = 1):
        for sentence in sentences:
            data = self.getWordWec(sentence["sentence"],verbouse)
            sentence["tokens"] = data[0]
            sentence["embedings"] = data[1]
        for sentence in sentences:
            dif = []
            for other in sentences:
                out = self.dist(sentence["embedings"][sentence["index"]],other["embedings"][other["index"]])
                dif.append(out)
            if verbouse > 0:
                print(sentence["sentence"],'\n',dif)

            
if __name__ == "__main__":
    embedings = Embedings()
    embedings.process("[CLS] Hello, this is Bert. [SEP]")