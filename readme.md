# Unicorn


## DataSets
We public 20 datasets of 7 matching tasks for Unicorn in "data/".
Each dataset contains train.json / valid.json /test.json. The details can be found in our paper.

- Entity Matching
    - em-wa: Walmart-Amazon
    - em-ds: DBLP-Scholar
    - em-fz: Fodors-Zagats
    - em-ia: iTunes-Amazon
    - em-beer: Beer
- Column Type Annotation
    - efthymiou: Efthymiou
    - t2d_col_type_anno: T2D
    - Limaye_col_type_anno: Limaye
- Entity Linking
    - t2d: T2D
    - Limaye: Limaye
- String Matching
    - smurf-addr: Address
    - smurf-names: Names
    - smurf-res: Researchers
    - smurf-prod: Product
    - smurf-cit: Citation
- Schema Matching
    - fabricated_dataset: FabricatedDatasets
    - DeepMDatasets: DeepMDatasets
- Ontology Matching
    - Illinois-onm: Cornell-Washington
- Entity Alignment
    - dbp_yg: SRPRS: DBP-YG
    - dbp_wd: SRPRS: DBP-WD


## Quick Start
Step 1: Requirements
- Before running the code, please make sure your Python version is 3.6.5 and cuda version is 11.1. Then install necessary packages by :
- `pip install -r requirements.txt`

Step 2: Run

Run the script for Unicorn:
-    `python main.py --pretrain --model deberta_base`

Run the script for Unicorn Imp:
-    `python main.py --pretrain --model deberta_base --shuffle 1 --load_balance 1`

Run the script for Unicorn Zero-shot:
-    `python main-zero.py --pretrain --model deberta_base`

Run the script for Unicorn Zero-shot Prompt:
-    `python main-zero-prompt.py --pretrain --model deberta_base`


