# Semantic parsing of requirements

Experiments on semantic parsing of technical requirements.

Note: You need requirement sentences extracted from PDF documents to conduct the experiments. In addition, the sentences must be annotated manually with DL formulas (F-prime) to do the evaluation. Tools to extract requirements from PDF documents and convert them into JSONL are found in the [req_extractor library](https://github.com/oholter/req_extractor).

As of June 2024, the documents used in the paper can be downloaded from DNV at https://www.veracity.com/.

# Preparations
Create and use a virtual environment:  
`python -m venv venv`  
`source ./venv/bin/activate`  

Install the requirements:  
`python -m pip install -r requirements.txt`

# Reproduce experiments

## Setup the experiment environment

1. Create a folder for the experiment:
`mkdir [experiment]`

2. Put a copy of the `config.json` file in the experiment folder:  
`cp gold_creator/config.json [experiment]`

3. Create empty files for the new gold standard and the order:  
``touch experiment.txt order.txt``

4. Create a *flat file* from a JSONL file with requirement sentences:  
``python -m utils.create_flat_file [JSONL file]``


5. Change the content of the copy of the `config.json` file to match the experiment setup you want. Note: You will need to input your API key from OpenAI in this file.


- Typically the flat file will be the "gold-gold" file (`gold.txt`) because this is where you take the samples from, this allows copying the F-primes.
* The "gold-gold" is a text file with requirement sentences. Each sentence in the flat file will have:
    * id: the number of the requirement, needed to manipulate the order of the requirements
    * req: the requirement sentence as extracted from the document
    * F-prime: the translation of the requirement into description logic, needed for evaluation. Added manually.
    * See ``data/gold.txt`` for an example
- The gold file and the order file are the empty files you created above (you're creating a new gold standard) and should be empty when you start a new experiment.
    - E.g., `experiment.txt` and ``order.txt``

## Running and evaluating the experiment


1. Run the experiments:  
`python -m gold_creator.runner_all --cfg [experiment]/config.json`

2. Calculate the scores:  
`python -m utils.calculate_measures [experiment]/[experiment.txt] -swjpeg`
The results are saved directly in the `experiment.txt` file

3. visualize the results:  
`python -m visualization.visualizer [experiment/experiment.txt] [edit|semantic|*] --order [order.txt] [--aggregate]`

4. Export the results to latex: prefix is a name prefix for the macro name:  
``python -m utils.experiment_to_latex_macro [experiment/experiment.txt] [PREFIX]``
