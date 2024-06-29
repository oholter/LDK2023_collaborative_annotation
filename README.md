# Semantic parsing of requirements

Experiments on semantic parsing of technical requirements

# Preparations
Create and use a virtual environment:
`python -m venv venv`  
`source ./venv/bin/activate`  

Install the requirements:  
`python -m pip install -r requirements.txt`

# Reproduce experiments:
Create a folder for the experiment:  
`mkdir [experiment]`
create a copy of the `config.json` file:  
`cp gold_creator/config.json [experiment]`
Change the content of the copy of the `config.json` file to match the experiment setup you want.
Note: You'll need to input your API key from OpenAI in this file

- Typically the flat file will be the gold-gold file (`gold.txt`) because this is where you take the samples from, this allows copying the F-primes
- The gold file and the order file empty (you're creating a new gold standard) whenever you start a new experiment.
    - e.g., `experiment.txt` and ``order.txt``

1. Run the experiments:
`python -m gold_creator.runner_all --cfg [experiment]/config.json`

2. Calculate the scores:
`python -m utils.calculate_measures [experiment]/[experiment.txt] -swjpeg`
The results are saved directly in the `experiment.txt`-file

3. visualize the results:
`python -m visualization.visualizer [experiment/experiment.txt] [edit|semantic|*] --order [order.txt] [--aggregate]`

4. Export the results to latex: prefix is a name prefix for the macro name
``python -m utils.experiment_to_latex_macro [experiment/experiment.txt] [PREFIX]``
