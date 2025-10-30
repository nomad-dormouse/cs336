# 6 Generating text

# Problem (decoding): Decoding (3 points)


from dotenv import load_dotenv, find_dotenv
import argparse
import re
import torch
from torch import Tensor
from jaxtyping import Float
import os
from tqdm import tqdm

from cs336_basics.tokeniser import Tokenizer
from cs336_basics.transformer import softmax
from cs336_basics.transformer import TransformerLM
from cs336_basics.train_transformer import get_default, get_device


load_dotenv(find_dotenv())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained language model")
    
    parser.add_argument("--device", type=str, default=get_default("device"), help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--model_filename", type=str, required=True, help="Name of the file that contains the model")
    parser.add_argument("--prompt", type=str, default=get_default("prompt"), help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=int(get_default("max_tokens")), help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=float(get_default("temperature")), help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=float(get_default("top_p")), help="Top-p sampling threshold")
    
    return parser.parse_args()


def convert_args(args: argparse.Namespace) -> dict:
    device = get_device(args.device)
    
    model_path = f"results/models/trained/best/{args.model_filename}.pt"

    name = args.model_filename.replace('.pt', '')
    # Updated pattern to include f (d_ff), r (learning_rate), and make trailing device optional (ignored)
    pattern = r'v(\d+)-c(\d+)-d(\d+)-f(\d+)-l(\d+)-h(\d+)-b(\d+)-r([\d.e-]+)-i(\d+)-(\w+)(?:-(\w+))?(?:-test)?'
    match = re.match(pattern, name)
    if not match:
        raise ValueError(
            f"Invalid model filename format: {args.model_filename}. Expected format: "
            f"v{vocab_size}-c{context_length}-d{d_model}-f{d_ff}-l{num_layers}-h{num_heads}-b{batch_size}-r{learning_rate}-i{max_iters}-{dataset}[-device][-test]"
        )
    vocab_size, context_length, d_model, d_ff, num_layers, num_heads, batch_size, learning_rate, max_iters, dataset, _ignored_device = match.groups()

    if dataset == "TS":
        vocab_size = 10000
        vocab_filepath = "results/tokeniser/vocab_TinyStoriesV2-GPT4-train_10000.json"
        merges_filepath = "results/tokeniser/merges_TinyStoriesV2-GPT4-train_10000.txt"
    elif dataset == "OWT":
        vocab_size = 32000
        vocab_filepath = "results/tokeniser/vocab_owt_train_32000.json"
        merges_filepath = "results/tokeniser/merges_owt_train_32000.txt"
    else:
        raise ValueError(f"Unknown dataset used to train the model: {dataset}. Expected: TS or OWT")
    
    return {
        'device': device,
        'model_path': model_path,
        'context_length': int(context_length),
        'num_layers': int(num_layers),
        'd_model': int(d_model),
        'd_ff': int(d_ff),
        'num_heads': int(num_heads),
        'batch_size': int(batch_size),
        'max_iters': int(max_iters),
        'vocab_size': vocab_size,
        'vocab_filepath': vocab_filepath,
        'merges_filepath': merges_filepath,
        'prompt': args.prompt,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
    }


def top_p_sampling(
    probs: Float[Tensor, "... vocab_size"],
    cumulative_probs_threshold: float,
) -> Float[Tensor, "... vocab_size"]:
    # Sort probabilities in descending order
    sorted_probs, sorted_probs_indices = torch.sort(probs, descending=True)
    
    # Compute the cumulative sum along the vocab dimension
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create a mask where cumulative prob exceeds threshold
    sorted_mask = cumulative_probs > cumulative_probs_threshold
    
    # Always keep at least one token (the top-1)
    sorted_mask[..., 0] = False
    
    # Map the sorted mask back to the original indices
    unsorted_mask = sorted_mask.scatter(-1, sorted_probs_indices, sorted_mask)

    # Zero out the probabilities that should be removed
    probs = probs.masked_fill(unsorted_mask, 0.0)
    
    # Renormalise the remaining probabilities to ensure they sum to 1
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return probs


def generate_text(
    device: str,
    model: TransformerLM,
    tokeniser: Tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    model.eval()

    eot_token = os.getenv("ENDOFTEXT_TOKEN").encode("utf-8")
    eot_token_id = tokeniser.token_to_id.get(eot_token)
    
    # Encode the prompt
    prompt_tokens = tokeniser.encode(prompt)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_tokens = prompt_tokens.copy()
    
    with torch.no_grad():
        for _ in tqdm(range(max_tokens), desc="Generating text"):
            # Get model output
            logits = model(prompt_tensor)
            
            # Get the last token's logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature scaling
            probs = softmax(next_token_logits, dim=-1, temperature=temperature)
            
            # Apply top-p sampling if specified
            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1).item()
            
            # Check for end-of-text token
            if next_token == eot_token_id:
                break
            
            # Add to generated tokens
            generated_tokens.append(next_token)
            
            # Update input for next iteration
            prompt_tensor = torch.cat([
                prompt_tensor, 
                torch.tensor([[next_token]], dtype=torch.long, device=device)
            ], dim=1)
    
    # Decode the generated text
    generated_text = tokeniser.decode(generated_tokens)

    return generated_text


def test_generation(
    args: argparse.Namespace,
) -> None:
    params = convert_args(args)
    print(f"Using device: {params['device']}")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(params['model_path'], map_location=params['device'])
    
    # Load tokeniser
    print("Loading tokeniser...")
    tokeniser = Tokenizer.from_files(
        vocab_filepath=params['vocab_filepath'],
        merges_filepath=params['merges_filepath']
    )
    
    # Create model with the same architecture as the trained one
    model = TransformerLM(
        vocab_size=params['vocab_size'],
        context_length=params['context_length'],
        num_layers=params['num_layers'],
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        d_ff=params['d_ff'],
    )
    state_dict = checkpoint['model_state_dict']
    # Handle models saved under torch.compile where keys are prefixed with '_orig_mod.'
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(params['device'])
    
    # Generate text
    print(f"\nPrompt: '{params['prompt']}'")
    print(f"Temperature: {params['temperature']}")
    print(f"Top-p: {params['top_p']}")
    
    generated_text = generate_text(
        device=params['device'],
        model=model,
        tokeniser=tokeniser,
        prompt=params['prompt'],
        max_tokens=params['max_tokens'],
        temperature=params['temperature'],
        top_p=params['top_p'],
    )
    print(f"\n{generated_text}")

    
if __name__ == "__main__":
    args = parse_args()
    test_generation(args)