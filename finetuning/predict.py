import logging
import json
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from finetuning.requirement_dataset import RequirementDataset, prepare_data
from utils.io import read_flat_file
from argparse import ArgumentParser
from pathlib import Path

MODEL_NAME = "facebook/bart-base"
CONFIG_FILE = "finetuning/config.json"


def predict(model, dataloader, config, device, data):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    total_loss = 0
    avg_val_loss = 0

    output_path = Path(config['pred']['output'])
    with output_path.open(mode='w') as out:
        out.write("req\n-->\tpred\n")
        logging.info("Predicting...")
        for i, (input_ids, input_attention_mask) in tqdm(enumerate(dataloader)):
            input_ids = input_ids.to(device)
            input_attention_mask = input_attention_mask.to(device)

            #output = model(input_ids=input_ids,
            #            attention_mask=input_attention_mask,
            #            return_dict=True)
            output = model.generate(input_ids=input_ids,
                                    max_length=256)
            #logits = output['logits']
            #pred_ids = logits.argmax(dim=2)
            pred_toks = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            print(pred_toks)
            out.write("{}\n --> {}\n".format(data[i]['req'], pred_toks))


def main():
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="config file")
    args = parser.parse_args()

    if args.cfg:
        config_file = args.cfg
    else:
        config_file = CONFIG_FILE

    with open(config_file, 'r') as F:
        config = json.load(F)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_path = Path(config['pred']['data'])
    model_path = Path(config['pred']['model'])
    data = read_flat_file(input_path, return_dict=True)
    num_test = config['pred']['max_samples']

    input_ids, input_attention_mask = prepare_data(data,
                                                   MODEL_NAME,
                                                   outputs=False)
    test_dataset = RequirementDataset(input_ids[:num_test],
                                      input_attention_mask[:num_test])

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False)

    model = torch.load(model_path)
    predict(model, test_dataloader, config, device, data)


if __name__ == '__main__':
    main()
