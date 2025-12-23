# XLdefgen [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/brandonwilde/XLdefgen)
### A framework for the automatic generation of cross-lingual definitions.

Read the full paper "Cross-Lingual Definition Generation from an mT5" included as a PDF in this repo.

### Abstract
Recent progress in definition modeling calls for an expansion into new territory. In this paper, we explore the little-studied prospect of cross-lingual definition generation from a resource-scarce perspective. We show that a small pretrained multilingual Text-to-Text Transfer Transformer (mT5) model can be transformed into a language-agnostic zero-shot definition generator, producing rudimentary English definitions for terms in multiple foreign languages. Throughout the project, several task-specific modifications to the model are devised and tested. We further recommend research paths that may progress the field of cross-lingual definition generation. 


## Data Access and Testing

Data should be accessed via the Hugging Face datasets library unless the desired dataset is already saved to disk.

Change the working directory to XLdefgen/model, and then run `run_model.py` as indicated in the model README.
