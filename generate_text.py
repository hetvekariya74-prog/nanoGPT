import torch
from nanoGPT import GPTLanguageModel, decode
import sys
import argparse


def main(model_path, max_new_tokens):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTLanguageModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[
        0
    ].tolist()
    generated_text = decode(generated_tokens)

    print(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a BigramLanguageModel."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--tokens", type=int, required=True, help="Number of tokens to generate."
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    main(args.model_path, args.tokens)
