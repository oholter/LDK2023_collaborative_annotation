import torch
from transformers import AutoTokenizer, AutoModel

#MODEL_NAME = "sentence-transformers/paraphrase-albert-small-v2"
MODEL_NAME = 'sentence-transformers/all-roberta-large-v1'


class SentenceVectorizer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model = self.model.to(self.device)

    def __call__(self, sent):
        return self.embed(sent)

    def tokenize(self, sent):
        #return self.tokenizer(sent, padding=True, truncation=True, max_length=128, return_tensors='pt')
        return self.tokenizer(sent, return_tensors='pt')

    def embed(self, sent):
        enc = self.tokenize(sent)
        enc = enc.to(self.device)
        model_output = self.model(**enc)
        token_embs = model_output[0]

        # mean pooling
        input_mask_expanded = enc['attention_mask'].unsqueeze(-1).expand(token_embs.size()).float()
        sum_embeddings = torch.sum(token_embs * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        sum = sum_embeddings / sum_mask
        sum = sum.cpu()

        return sum.squeeze().detach().numpy()





def main():
    vectorizer = SentenceVectorizer()
    output = vectorizer("This is a sentence")
    print(output)
    print(output.shape)

if __name__ == '__main__':
    main()
