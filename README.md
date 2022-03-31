# ChallengeSet_SemanticRoleLabeling
Author: YC Roderick Li

This project uses CHECKLIST (https://github.com/marcotcr/checklist) to create a linguistically-motivated challenge set for semantic role labeling systems and to test two pretrained SRL predictors from AllenNLP (https://github.com/allenai/allennlp). 

## Step 1
run `create_dataset.py`. A .json file consisting the test instances will be saved in the `/data` folder.

The challenge set tests the system on 5 capabilities:
1. Be disambiguation: to differentiate the copula 'be' and the auxilliary 'be' and correctly classify the argument.
2. Location recognition: to recognize location in the 'work in' construction and classify the argument as ARGM-LOC.
3. ARG1 negation: a simple test on the system's basic capability to classify a negated ARG1.
4. Causative alternation: to correctly detect the ARG1 in transitive/intransitive verb alternation.
5. Passive voice: a simple test on finding the ARG1 in active and passive constructions.

## Step 2
run `run_tests.py`. Two .json files containing test results from the two models will be saved in the `/output` folder.

The test type for each capability is documented as follows:
1. Be disambiguation: DIR
2. Location recognition: DIR
3. ARG1 negation: MFT
4. Causative alternation: INV
5. Passive voice: INV
