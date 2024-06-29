from transformers import AutoTokenizer
from torch.utils.data import Dataset


class RequirementDataset(Dataset):
    def __init__(self,
                 input_ids,
                 input_attention_mask,
                 output_ids=None,
                 output_attention_mask=None):
        self.input_ids = input_ids
        self.input_attention_mask = input_attention_mask
        self.gold_ids = output_ids
        self.gold_attention_mask = output_attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if (self.gold_ids is not None) and (self.gold_attention_mask is not None):
            return (self.input_ids[idx],
                    self.input_attention_mask[idx],
                    self.gold_ids[idx],
                    self.gold_attention_mask[idx])
        elif self.gold_ids:
            return (self.input_ids[idx],
                    self.input_attention_mask[idx],
                    self.gold_ids[idx])
        else:
            return self.input_ids[idx], self.input_attention_mask[idx]


def prepare_data(data, model_name, outputs=True):
    """
    data: a nested dict {"id": {"req": "...", "F-prime": "..."}}
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = []
    attention_masks = []
    labels = []

    inputs = [r['req'] for r in data.values()]
    encoded_input = tokenizer.batch_encode_plus(inputs,
                                                return_tensors='pt',
                                                padding=True,
                                                truncation=True)

    if outputs:
        outputs = [r['F-prime'] for r in data.values()]
        encoded_output = tokenizer.batch_encode_plus(outputs,
                                                     return_tensors='pt',
                                                     padding=True,
                                                     truncation=True)
        return (encoded_input['input_ids'],
                encoded_input['attention_mask'],
                encoded_output['input_ids'],
                encoded_output['attention_mask'])
    else:
        return encoded_input['input_ids'], encoded_input['attention_mask']
