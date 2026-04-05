# nanoGPT: Decoder only Transformer model for text generation

This repository contains the code for the nanoGPT model, a lightweight version of the GPT model. The model is trained on the all Williams Shakespeare's works and can be used for generate text in the style of Shakespeare.

This is decoder only transformer model, which means it only contains the decoder part of the transformer model. All the code is has been implemented in PyTorch from scratch.

## Usage

To use the model, you first have to download the pre-trained model from the following link:

[Download nanoGPT model](https://drive.google.com/file/d/1u_wkiQz_f40eOus61ijRHWlpmW-1Hxss/view?usp=sharing)

After downloading the model, you can generate text using the following command:

```bash
python generate_text.py --model_path models/nanoGPT.pth --tokens 100
```

NOTE:
- `bigram_v1.py` is the first version of the bigram model.
- `bigram_v2.py` is the second version of the bigram model with single head self-attention
- `bigram_v3.py` is the third version of the bigram model with multi-head self-attention and residual connections.
- `nanoGPT.py` is the final version of the model with multi-head self-attention, residual connections, and some regularization techniques