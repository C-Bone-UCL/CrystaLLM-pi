# Tutorial Notebooks

Five notebooks in [`notebooks/`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/) cover the maintained workflows end to end:

* [`T1_finetune_density_example.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T1_finetune_density_example.ipynb): finetune a base model on your own property dataset, push it to the Hub, register it, and generate with it
* [`T2_load_and_generate.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T2_load_and_generate.ipynb): generate structures with the released Hub models (courtesy of [Joley Lin](https://github.com/yhjollin/))
* [`T3_API_density_example.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T3_API_density_example.ipynb): predict density for a composition through the containerised API
* [`T4_XRD_continuous.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T4_XRD_continuous.ipynb): recover a structure from a raw experimental XRD scan with the continuous-XRD model
* [`T5_SLME.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T5_SLME.ipynb): discover a material with a target photovoltaic efficiency

The paper studies are not here, they live in [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper).

## Customising the tokenizer

The `HF-cif-tokenizer` already contains everything needed to train and run the models. To add more tokens:

* **Create the new vocab**: Edit the [`create_vocab.py`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_utils/_tokenizer/create_vocab.py) file to include all the new tokens you want (if augmenting CIF with new tokens for example). Save a new `vocabulary.json` with the updated dictionary.
* **Optional: Add Spacegroups**: If new spacegroups are required for a particular study, these should be added to the [`spacegroups.txt`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_utils/_tokenizer/spacegroups.txt) file.
* **Build New Tokenizer**: Once the new vocabulary is ready, run the [`save_tokenizer_to_hf.py`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_utils/_preprocessing/save_tokenizer_to_hf.py) script, to save it locally or to HF. Give it its own `--path`, since the default overwrites the tokenizer that ships with the package. Then point the `pretrained_tokenizer_dir` argument in the train config at your new tokenizer.
