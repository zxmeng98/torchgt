def transformer_memory_estimate(batch_size, input_dim, hidden_dim, num_heads, num_layers, seq_length, ffn_size):
    attn_dim = hidden_dim / num_heads 
    
    # Input graph features
    input_mem = batch_size * seq_length * input_dim * 4  # bytes
    print("Input mem: {}MB".format(input_mem / (1024 * 1024)))
    
    # Model states: weights, gradients, optimizer states (momentum + variance)
    model_mem = num_layers * (3 + 1 + ffn_size + ffn_size) * hidden_dim * hidden_dim * 24 # bytes
    print("Model states: {}MB".format(model_mem / (1024 * 1024)))

    # Activations: fw + bw
    acti_mem = num_layers * (
        (batch_size * seq_length * hidden_dim) + \
        (num_heads * 3 * batch_size * seq_length * attn_dim) + \
        (num_heads * 3 * batch_size * seq_length * seq_length) + \
        (num_heads * batch_size * seq_length * attn_dim) + \
        (batch_size * seq_length * hidden_dim) + \
        (batch_size * seq_length * hidden_dim) + \
        (batch_size * seq_length * ffn_size * hidden_dim) + \
        (batch_size * seq_length * hidden_dim) + \
        (batch_size * seq_length * hidden_dim)
        ) + \
        max(
        (batch_size * seq_length * hidden_dim),
        (num_heads * 3 * batch_size * seq_length * attn_dim),
        (num_heads * 3 * batch_size * seq_length * seq_length),
        (num_heads * batch_size * seq_length * attn_dim),
        (batch_size * seq_length * hidden_dim),
        (batch_size * seq_length * hidden_dim),
        (batch_size * seq_length * ffn_size * hidden_dim),
        (batch_size * seq_length * hidden_dim),
        (batch_size * seq_length * hidden_dim)
        ) 
    acti_mem *= 4  # bytes
    print("Activation: {}MB".format(acti_mem / (1024 * 1024)))

    # # Graph-specific encoding, no edge encoding now
    # ge_num = attn_weights_num
    # print("Graph Encoding: {}MB".format(ge_num * 4 / (1024 * 1024)))

    # Total memory in MB 
    total_mem_mb = (input_mem + model_mem + acti_mem ) / (1024 * 1024)
    print(f"Estimated memory usage for the Transformer model: {total_mem_mb:.2f} MB")
    
    return total_mem_mb

# Example usage:
num_heads = 8
num_layers = 4
seq_length = 4000
batch_size = 1
input_dim = 602
hidden_dim = 128
ffn_size = 4

estimated_mem = transformer_memory_estimate(batch_size, input_dim, hidden_dim, num_heads, num_layers, seq_length, ffn_size)
