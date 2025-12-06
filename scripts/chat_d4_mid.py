import os
import torch

from reqchat.checkpoint_manager import load_model_from_dir
from reqchat.tokenizer import get_tokenizer

# --- config ---
DEVICE = torch.device("cpu")  # or "cuda" if you have a GPU
# Base dir for mid checkpoints (parent of d4/)
MID_CHECKPOINTS_DIR = os.path.expanduser(r"C:\Users\mohan\.cache\reqchat\mid_checkpoints")
MODEL_TAG = "d4"
MAX_NEW_TOKENS = 32
MAX_CTX_TOKENS = 64  # your d4 was trained with sequence_len=64

def load_model_and_tokenizer():
    # phase must be "train" or "eval"; we want inference, so use "eval"
    out = load_model_from_dir(
        MID_CHECKPOINTS_DIR,
        phase="eval",
        device=DEVICE,
        model_tag=MODEL_TAG,
    )
    # load_model_from_dir may return (model, tokenizer, meta, ...) or similar
    if isinstance(out, tuple):
        model = out[0]
    else:
        model = out

    model.to(DEVICE)
    model.eval()

    tokenizer = get_tokenizer()
    return model, tokenizer


@torch.no_grad()
def generate_reply(model, tokenizer, prompt,
                   max_new_tokens=MAX_NEW_TOKENS,
                   max_ctx_tokens=MAX_CTX_TOKENS):
    # Encode prompt
    input_ids = tokenizer.encode(prompt)

    # Ensure we don't exceed context window
    if len(input_ids) > max_ctx_tokens:
        input_ids = input_ids[-max_ctx_tokens:]

    x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)[None, :]  # (1, T)

    for _ in range(max_new_tokens):
        # Make sure we don't grow beyond context window
        if x.size(1) >= max_ctx_tokens:
            break

        logits = model(x)  # reqchat GPT forward(x, ...) returns logits
        if isinstance(logits, tuple):
            logits = logits[0]

        logits_last = logits[:, -1, :]  # (1, vocab)
        probs = torch.softmax(logits_last, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)  # greedy

        x = torch.cat([x, next_token], dim=1)

    out_tokens = x[0].tolist()
    text = tokenizer.decode(out_tokens)
    return text

def main():
    print("Loading d4 mid-trained model (phase='mid', tag='d4') from base dir:")
    print(" ", MID_CHECKPOINTS_DIR)
    model, tokenizer = load_model_and_tokenizer()
    print("Loaded. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user.strip().lower() in {"quit", "exit"}:
            print("Bye!")
            break

        prompt = f"USER: {user}\nASSISTANT:"
        reply = generate_reply(model, tokenizer, prompt)

        print("Model:", reply)
        print()

if __name__ == "__main__":
    main()