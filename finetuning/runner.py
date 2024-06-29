import torch
import logging
import json
import math
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
#from transformers import AutoModelForSeq2SeqLM
from transformers import BartForConditionalGeneration
#from transformers import AutoModel
from torch.utils.data import DataLoader
from utils.io import read_flat_file, remove_ignore, remove_missing_fprime
from finetuning.requirement_dataset import RequirementDataset, prepare_data

MODEL_NAME = "facebook/bart-base"
#MODEL_NAME = "t5-base"
CONFIG_FILE = "finetuning/config.json"


def fine_tune_bart(model,
                   train_dataloader,
                   validation_dataloader,
                   config, device="cpu"):
    num_epochs = config['parameters']['epochs']
    lr = config['parameters']['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        logging.info("Epoch: {}".format(epoch))
        model.train()
        total_loss = 0
        for input_ids, input_attention_mask, gold_ids, gold_attention_mask in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            input_attention_mask = input_attention_mask.to(device)
            gold_ids = gold_ids.to(device)
            gold_attention_mask = gold_attention_mask.to(device)

            optimizer.zero_grad()
            loss = model(input_ids, 
                    attention_mask=input_attention_mask, 
                    #decoder_input_ids=gold_ids,
                    #decoder_attention_mask=gold_attention_mask,
                    labels=gold_ids)[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        total_loss = 0
        avg_val_loss = 0
        logging.info("Evaluation")
        for input_ids, input_attention_mask, gold_ids, gold_attention_mask in tqdm(validation_dataloader):
            input_ids = input_ids.to(device)
            input_attention_mask = input_attention_mask.to(device)
            gold_ids = gold_ids.to(device)
            gold_attention_mask = gold_attention_mask.to(device)

            loss = model(input_ids=input_ids,
                         attention_mask=input_attention_mask,
                         #decoder_input_ids=gold_ids,
                         #decoder_attention_mask=gold_attention_mask,
                         labels=gold_ids,
                         return_dict=False)[0]
            #print(output)
            #exit()

            total_loss += loss.item()
            avg_val_loss = total_loss / len(validation_dataloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, config['io']['best_model'])
            logging.info("Saving current model to: {}".format(config['io']['best_model']))

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


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

    input_path = Path(config['io']['input'])
    data = read_flat_file(input_path, return_dict=True)
    remove_ignore(data)
    remove_missing_fprime(data)

    input_ids, input_attention_mask, output_ids, output_attention_mask = prepare_data(data, MODEL_NAME)
    train_samples = math.floor((1-config['parameters']['test_size']) * len(input_ids))
    train_samples = min(train_samples, config['parameters']['max_train_samples'])

    train_dataset = RequirementDataset(input_ids[:train_samples],
                                       input_attention_mask[:train_samples],
                                       output_ids[:train_samples],
                                       output_attention_mask[:train_samples])

    val_dataset = RequirementDataset(input_ids[train_samples:],
                                     input_attention_mask[train_samples:],
                                     output_ids[train_samples:],
                                     output_attention_mask[train_samples:])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['parameters']['batch_size'],
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['parameters']['batch_size'],
                                shuffle=False)

    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    fine_tune_bart(model,
                   train_dataloader,
                   val_dataloader,
                   config=config,
                   device=device)

    logging.info("Saving last model to: {}".format(config['io']['last_model']))
    torch.save(model, config['io']['last_model'])


if __name__ == '__main__':
    main()
