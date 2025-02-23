import pytorch_lightning as pl
from more_itertools import chunked

from transformers import AutoTokenizer
from transformers import AutoModel

import torch

#MODEL_NAME = "sentence-transformers/stsb-xlm-r-multilingual"
MODEL_NAME = "sentence-transformers/paraphrase-albert-small-v2"

class LMCosineSim(pl.LightningModule):
    def __init__(self, bs=16):
        super().__init__()
        # following the setup from here: https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                                       add_special_tokens=False)

        # Loading the model 2 times in order to be able to use integrated gradients to manipulate its embeddings
        self.hyp_model = AutoModel.from_pretrained(MODEL_NAME)
        self.src_model = AutoModel.from_pretrained(MODEL_NAME)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.bs = bs
        self.version = "custom"

    def __call__(self, src, hyp):
        '''
        :param ref: A list of strings with source sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of Labse Scores
        '''
        self.eval()

        src_ids = self.tokenize(src)
        hyp_ids = self.tokenize(hyp)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        scores = []

        # Unfortunately I didn't find existing batching here (so I'd suppose it should be there)
        # Therefore I use some other lib for batching
        for x in chunked(range(src_ids['input_ids'].shape[0]), self.bs):
            s_in_batch = src_ids['input_ids'][x]
            s_in_batch = s_in_batch.to(device)
            h_in_batch = hyp_ids['input_ids'][x]
            h_in_batch = h_in_batch.to(device)
            s_att_batch = src_ids['attention_mask'][x]
            s_att_batch = s_att_batch.to(device)
            h_att_batch = hyp_ids['attention_mask'][x]
            h_att_batch = h_att_batch.to(device)

            scores += self.forward(s_in_batch, h_in_batch,
                               s_att_batch, h_att_batch).tolist()

        return scores

    def tokenize(self, sent):
        return self.tokenizer(sent, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        # pooling step from here: https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, src_ids, hyp_ids, src_mask, hyp_mask):
        src_model_out = self.hyp_model(input_ids=src_ids, attention_mask=src_mask)
        src_emb = self.mean_pooling(src_model_out, src_mask)

        hyp_model_out = self.src_model(input_ids=hyp_ids, attention_mask=hyp_mask)
        hyp_emb = self.mean_pooling(hyp_model_out, hyp_mask)
        cos_sim = self.cos(src_emb, hyp_emb)

        return cos_sim


if __name__ == '__main__':
    c = LMCosineSim()

    # Sample using ref and hyp lists
    # [0.9784649610519409]
    print(c(["Ein Test Satz"], ["A test sentence"]))
    print(c(["Test"], ["test"]))
