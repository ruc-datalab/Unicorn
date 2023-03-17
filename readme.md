# Unicorn

![python](https://img.shields.io/badge/python-3.6.5-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.7.1-brightgreen)

<img src="figs/matching-tasks.png" width="820" />

Data matching – which decides whether two data elements (e.g., string, tuple, column, or knowledge graph entity) are the “same” (a.k.a. a match) – is a key concept in data integration, such as entity matching and schema matching. The widely used practice is to build task-specific or even dataset-specific solutions, which are hard to generalize and disable the opportunities of knowledge sharing that can be learned from different datasets and multiple tasks. 
In this paper, we propose Unicorn, a unified model for generally supporting common data matching tasks. Moreover, this unified model can enable knowledge sharing by learning from multiple tasks and multiple datasets, and can also support zero-shot prediction for new tasks with zero labeled matching/non-matching pairs. However, building such a unified model is challenging due to heterogeneous formats of input data elements (e.g., strings, tuples, columns, trees, graphs, and so on) and various matching semantics of multiple tasks. To address the challenges, Unicorn employs one generic Encoder that converts any pair of data elements () into a learned representation, and uses a Matcher, which is a binary classifier, to decide whether matches . To align matching semantics of multiple tasks, Unicorn also adopts a mixture-of-experts model that enhances the learned representation into a better representation, which can further boost the performance of predictions. We conduct extensive experiments on 20 datasets of seven well studied data matching tasks, including entity matching, entity linking, entity alignment, column type annotation, string matching, schema matching, and ontology matching, and find that our unified model can achieve better performance on most tasks and on average, compared with the state-of-the-art specific models trained for ad-hoc tasks and datasets separately. Moreover, Unicorn can also well serve new matching tasks with zero-shot learning. 


## DataSets
We publish 20 datasets of 7 matching tasks in Unicorn.
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

<img src="figs/framework.png" width="820" />


## Quick Start
Step 1: Requirements
- Before running the code, please make sure your Python version is 3.6.5 and cuda version is 11.1. Then install necessary packages by :
- `pip install -r requirements.txt`
- `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

Step 2: Run

Run the script for Unicorn:
-    `cuda_visible_devices=1 python main.py --pretrain --model deberta_base`

Run the script for Unicorn ++:
-    `cuda_visible_devices=1 python main.py --pretrain --model deberta_base --shuffle 1 --load_balance 1`

Run the script for Unicorn Zero-shot:
-    `cuda_visible_devices=1 python main-zero.py --pretrain --model deberta_base`

Run the script for Unicorn Zero-shot Prompt:
-    `cuda_visible_devices=1 python main-zero-prompt.py --pretrain --model deberta_base`

Finetune model with your dataset:
-    `cuda_visible_devices=1 python finetune.py --load --namef UnicornPlus --model deberta_base --train_dataset_path "train_file_path.json" --valid_dataset_path "valid_file_path.json" --test_dataset_path "rest_file_path.json" `

Load model and direct test: 
-    `cuda_visible_devices=1 python test.py --load --namef UnicornPlus --model deberta_base --dataset_path "test_file_path1.json test_file_path2.json ..."`
