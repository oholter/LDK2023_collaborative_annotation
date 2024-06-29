import logging
import json
import torch
import math
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from finetuning.requirement_dataset import RequirementDataset, prepare_data
from utils.io import read_flat_file, remove_ignore, remove_missing_fprime
from argparse import ArgumentParser
from pathlib import Path

MODEL_NAME = "facebook/bart-base"
CONFIG_FILE = "finetuning/config.json"


def test_model(model, dataloader, config, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    total_loss = 0
    avg_val_loss = 0

    output_path = Path(config['test']['output'])
    with output_path.open(mode='w') as out:
        out.write("gold\n-->\tpred\n")
        logging.info("Testing...")
        for input_ids, input_attention_mask, gold_ids, gold_attention_mask in tqdm(dataloader):
            input_ids = input_ids.to(device)
            input_attention_mask = input_attention_mask.to(device)
            gold_ids = gold_ids.to(device)
            gold_attention_mask = gold_attention_mask.to(device)

            output = model.generate(input_ids=input_ids,
                                    max_length=256)
                                    #do_sample=True,
                                    #num_beams=5,
                                    #early_stopping=True)
                        #attention_mask=input_attention_mask,
                        #decoder_input_ids=gold_ids,
                        #decoder_attention_mask=gold_attention_mask,
                        #labels=gold_ids,
                        #return_dict=True)
            #loss = output['loss']
            #logits = output['logits']
            #print(output)
            gold_toks = tokenizer.batch_decode(gold_ids, skip_special_tokens=True)[0]
            #pred_ids = logits.argmax(dim=2)
            #pred_toks = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            pred_toks = tokenizer.batch_decode(output)
            print(pred_toks)
            out.write("{}\n --> {}\n".format(gold_toks, pred_toks))


            #total_loss += loss.item()
            #avg_val_loss = total_loss / len(dataloader)


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

    input_path = Path(config['test']['data'])
    model_path = Path(config['test']['model'])
    data = read_flat_file(input_path, return_dict=True)
    remove_ignore(data)
    remove_missing_fprime(data)
    num_test = config['test']['max_test_samples']

    input_ids, input_attention_mask, output_ids, output_attention_mask = prepare_data(data, MODEL_NAME)

    test_dataset = RequirementDataset(input_ids[:num_test],
                                      input_attention_mask[:num_test],
                                      output_ids[:num_test],
                                      output_attention_mask[:num_test])

    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False)

    model = torch.load(model_path)

    test_model(model, test_dataloader, config, device)



if __name__ == '__main__':
    main()
