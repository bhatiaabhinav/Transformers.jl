using Flux

"""
    ResidualAndNorm(sublayer, dim::Int; dropout=0.0)

Residual connection and layer normalization for Transformer. It is composed of a sublayer and a layer normalization layer. Optionally, a dropout layer can be added. The output of the sublayer is assumed to be the same size as the input to the residual connection. When calling the layer, the residual input and inputs to the sublayer are required seperately.

# Arguments
- `sublayer`: sublayer
- `dim`: output dimension of the sublayer
- `dropout`: dropout probability

# Example
```julia
using Flux

sublayer = Dense(20, 10)
r = ResidualAndNorm(sublayer, 10)
r_input = randn(Float32, 10, 32)
sublayer_input = randn(Float32, 20, 32)
r(r_input, sublayer_input, optional_sublayer_input_args...) # size (10, 32)
```
===
"""
struct ResidualAndNorm
    sublayer
    layernorm::LayerNorm
    dropout::Union{Dropout, Nothing}
end
Flux.@functor ResidualAndNorm

function ResidualAndNorm(sublayer, dim::Int; dropout=0.0)
    layernorm = LayerNorm(dim)
    if dropout > 0.0
        dropout = Dropout(dropout)
    else
        dropout = nothing
    end
    return ResidualAndNorm(sublayer, layernorm, dropout)
end

"""
    (r::ResidualAndNorm)(residual_input, sublayer_inputs...)

# Arguments
- `residual_input`: input to the residual connection
- `sublayer_inputs`: inputs to the sublayer
"""
function (r::ResidualAndNorm)(residual_input, sublayer_inputs...)
    x = r.layernorm(r.sublayer(sublayer_inputs...) + residual_input)
    if r.dropout !== nothing
        x = r.dropout(x)
    end
    return x
end





"""
    EncoderLayer(dim::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_ff::Int; dropout=0.0, σ=gelu, linear_attention=false)

Encoder layer for Transformer. It is composed of a multi-head self-attention sublayer followed by a feedforward sublayer. Each sublayer is wrapped with a residual connection, a layer normalization layer and optional dropout. Therefore the size of the input and output of the encoder layer is the same.

# Arguments
- `dim`: input and output dimension of the encoder layer
- `dim_k`: output dimension of query and key layers in each attention head
- `dim_v`: output dimension of value layer in each attention head
- `num_heads`: number of attention heads
- `dim_ff`: hidden dimension of the feedforward sublayer
- `dropout`: dropout probability for each sublayer
- `σ`: activation function for the feedforward sublayer (default: `gelu`)
- `linear_attention`: if `true`, use linear attention instead of dense scaled dot-product attention

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236.pdf)

# Example
```julia
layer = EncoderLayer(512, 64, 64, 8, 2048);
seq_len = 20
batch_size = 32
x = randn(Float32, 512, seq_len, batch_size);
ouput = layer(x); # size (512, seq_len, batch_size)
@assert size(output) == (512, seq_len, batch_size)
```
===
"""
struct EncoderLayer
    attn::ResidualAndNorm
    feedforward::ResidualAndNorm
end

Flux.@functor EncoderLayer

function EncoderLayer(dim::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_ff::Int; dropout=0.0, σ=gelu, linear_attention=false)
    MHSA_Cls = linear_attention ? MultiHeadLinearSelfAttention : MultiHeadSelfAttention
    attn = MHSA_Cls(dim, dim_k, dim_v, num_heads, dim, false)
    attn = ResidualAndNorm(attn, dim; dropout=dropout)
    feedforward = Chain(Dense(dim, dim_ff, σ), Dense(dim_ff, dim))
    feedforward = ResidualAndNorm(feedforward, dim; dropout=dropout)
    return EncoderLayer(attn, feedforward)
end

"""
    (layer::EncoderLayer)(x)

# Arguments
- `x`: input of shape (dim, seq_len_enc, batch_size)
"""
function (layer::EncoderLayer)(x)
    x = layer.attn(x, x) # self-attention with query, key, value all being x. output shape = (dim_v, seq_len_enc, batch_size)
    x = layer.feedforward(x, x) # output shape = (dim_ff, seq_len_enc, batch_size)
    return x
end


"""
    Encoder(dim::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_ff::Int, num_layers::Int; dropout=0.0, σ=gelu, linear_attention=false)

Encoder for Transformer. It is composed of a stack of encoder layers and the final output is the output of the last encoder layer. The input is assumed to be already embedded and positional encoded.

# Arguments
- `dim`: input and output dimension of the encoder
- `dim_k`: output dimension of query and key layers in each attention head
- `dim_v`: output dimension of value layer in each attention head
- `num_heads`: number of attention heads
- `dim_ff`: hidden dimension of the feedforward sublayer
- `num_layers`: number of encoder layers
- `dropout`: dropout probability for each sublayer
- `σ`: activation function for the feedforward sublayer in each encoder layer (default: `gelu`)
- `linear_attention`: if `true`, use linear attention instead of dense scaled dot-product attention

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236.pdf)

# Example
```julia
encoder = Encoder(512, 64, 64, 8, 2048, 6);
seq_len = 20
batch_size = 32
src = randn(Float32, 512, seq_len, batch_size);  # input is assumed to be already embedded and positional encoded
enc_out = encoder(src) # size (512, seq_len, batch_size);
@assert size(enc_out) == (512, seq_len, batch_size)
```

"""
struct Encoder
    layers::Vector{EncoderLayer}
end
Flux.@functor Encoder

function Encoder(dim::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_ff::Int, num_layers::Int; dropout=0.0, σ=gelu, linear_attention=false)
    layers = [EncoderLayer(dim, dim_k, dim_v, num_heads, dim_ff; dropout=dropout, σ=σ, linear_attention=linear_attention) for _ in 1:num_layers]
    return Encoder(layers)
end

"""
    (encoder::Encoder)(src)

# Arguments
- `src`: input of shape (dim, seq_len_enc, batch_size)

# Returns
- output of the last encoder layer of shape (dim, seq_len_enc, batch_size)

Note: The `src` input is assumed to be already embedded and positional encoded.
"""
function (encoder::Encoder)(src)
    x = src
    for layer in encoder.layers
        x = layer(x)
    end
    return x
end





"""
    DecoderLayer(dim::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_ff::Int; dropout=0.0, σ=gelu, no_encoder=false, linear_attention=false)

Decoder layer for Transformer. It is composed of a masked multi-head self-attention attention sublayer, a multi-head encoder-decoder attention sublayer that accepts key and value input from the encoder (potentially of different seq_len) while accepting query input from the first sublayer, and a feedforward sublayer. If no encoder output is provided during inference, then the encoder-decoder attention sublayer is ignored. Each sublayer is wrapped with a residual connection, a layer normalization layer and optional dropout. The output of the decoder layer has the same size and sequence length as the output of the previous decoder layer.

# Arguments
- `dim`: input and output dimension of the decoder layer
- `dim_k`: output dimension of query and key layers in each attention head. Usually `dim_k = dim_v = dim / num_heads`
- `dim_v`: output dimension of value layer in each attention head
- `num_heads`: number of attention heads
- `dim_ff`: hidden dimension of the feedforward sublayer. Usually `dim_ff = 4 * dim`
- `dropout`: dropout probability for each sublayer
- `σ`: activation function for the feedforward sublayer (default: `gelu`)
- `no_encoder`: if `true`, ignore the encoder-decoder attention sublayer. This is useful when no encoder output is provided.
- `linear_attention`: if `true`, use linear attention instead of dense scaled dot-product attention

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236.pdf)

# Example
```julia
decoder_layer = DecoderLayer(512, 64, 64, 8, 2048);
src_seq_len = 10  # encoder sequence length
batch_size = 32
enc_out = randn(Float32, 512, src_seq_len, batch_size);  # encoder output
seq_len = 20
x = randn(Float32, 512, seq_len, batch_size);
output = decoder_layer(x, enc_out);   # size (512, seq_len, batch_size)
@assert size(output) == (512, seq_len, batch_size)
# or if no encoder output is provided. `no_encoder` must be set to true
output = decoder_layer(x);            # size (512, seq_len, batch_size)
@assert size(output) == (512, seq_len, batch_size)
```
===
"""
mutable struct DecoderLayer
    attn1::ResidualAndNorm
    attn2::Union{ResidualAndNorm, Nothing} # if `no_encoder`, then this is set to nothing
    feedforward::ResidualAndNorm
    cache
end
Flux.@functor DecoderLayer (attn1, attn2, feedforward)

function DecoderLayer(dim::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_ff::Int; dropout=0.0, σ=gelu, no_encoder=false, linear_attention=false)
    MHSA_Cls = linear_attention ? MultiHeadLinearSelfAttention : MultiHeadSelfAttention
    attn1 = MHSA_Cls(dim, dim_k, dim_v, num_heads, dim, true)
    attn1 = ResidualAndNorm(attn1, dim; dropout=dropout)
    if !no_encoder
        attn2 = MHSA_Cls(dim, dim_k, dim_v, num_heads, dim, false)
        attn2 = ResidualAndNorm(attn2, dim; dropout=dropout)
    else
        attn2 = nothing
    end
    feedforward = Chain(Dense(dim, dim_ff, σ), Dense(dim_ff, dim))
    feedforward = ResidualAndNorm(feedforward, dim; dropout=dropout)
    return DecoderLayer(attn1, attn2, feedforward, nothing)
end

"""
    (layer::DecoderLayer)(x, enc_out)

# Arguments
- `x`: input to the decoder layer of shape (dim, seq_len_dec, batch_size)
- `enc_out`: output of the encoder of shape (dim, seq_len_enc, batch_size)
"""
function (layer::DecoderLayer)(x, enc_out)
    _x = layer.attn1(x, x) # self-attention, x is the query, key and value. output shape: (dim_v, seq_len_dec, batch_size)
    _x = layer.attn2(_x, _x, enc_out, enc_out) # encoder-decoder attention, _x is the query. enc_out is the key and value. output shape: (dim_v, seq_len_dec, batch_size)

    L = size(x, 2)
    incremental = !haskey(ENV, "DISABLE_INCREMENTAL_ATTENTION") || ENV["DISABLE_INCREMENTAL_ATTENTION"] != "true"
    if incremental && layer.cache === nothing
        incremental=false
    end
    if incremental
        prev_x, _x_old = layer.cache
        if size(prev_x, 2) != L - 1 || !(sum(selectdim(prev_x, 2, 1)) ≈ sum(selectdim(x, 2, 1))) || !(sum(selectdim(prev_x, 2, L-1)) ≈ sum(selectdim(x, 2, L-1)))
            incremental = false
        end
    end
    if incremental
        _x_new = selectdim(_x, 2, L:L) |> copy # TODO: do we need to create copies?
        _x_new = layer.feedforward(_x_new, _x_new)     # feedforward layer. output shape: (dim_ff, 1, batch_size)
        _x = cat(_x_old, _x_new, dims=2) # shape: (dim_ff, seq_len_dec, batch_size)
    else
        _x = layer.feedforward(_x, _x)     # feedforward layer. output shape: (dim_ff, seq_len_dec, batch_size)
    end
    layer.cache = copy.((x, _x))
    return _x
end

"""
    (layer::DecoderLayer)(x)

    Ignores the encoder output and performs self-attention only i.e., the encoder-decoder attention sublayer is ignored.

# Arguments
- `x`: input to the decoder layer of shape (dim, seq_len_dec, batch_size)
"""
function (layer::DecoderLayer)(x)
    _x = layer.attn1(x, x) # self-attention, x is the query, key and value. output shape: (dim_v, seq_len_dec, batch_size)
 
    L = size(x, 2)
    incremental = !haskey(ENV, "DISABLE_INCREMENTAL_ATTENTION") || ENV["DISABLE_INCREMENTAL_ATTENTION"] != "true"
    if incremental && layer.cache === nothing
        incremental=false
    end
    if incremental
        prev_x, _x_old = layer.cache
        if size(prev_x, 2) != L - 1 || !(sum(selectdim(prev_x, 2, 1)) ≈ sum(selectdim(x, 2, 1))) || !(sum(selectdim(prev_x, 2, L-1)) ≈ sum(selectdim(x, 2, L-1)))
            incremental = false
        end
    end
    if incremental
        _x_new = selectdim(_x, 2, L:L) |> copy
        _x_new = layer.feedforward(_x_new, _x_new)     # feedforward layer. output shape: (dim_ff, 1, batch_size)
        _x = cat(_x_old, _x_new, dims=2) # shape: (dim_ff, seq_len_dec, batch_size)
    else
        _x = layer.feedforward(_x, _x)     # feedforward layer. output shape: (dim_ff, seq_len_dec, batch_size)
    end
    layer.cache = (copy(x), _x)
    return _x
end




"""
    Decoder(dim::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_ff::Int, num_layers::Int; dropout=0.0, σ=gelu, no_encoder=false, linear_attention=false)

Decoder for Transformer. It is composed of a stack of decoder layers and the output is the output of the last decoder layer. The input is assumed to be already embedded and positional encoded. The decoder also needs the output of the encoder to perform encoder-decoder attention. If no encoder output is provided during inference, then each decoder layer performs self-attention only and ignores the encoder-decoder attention sublayer.

# Arguments
- `dim`: input and output dimension of the decoder
- `dim_k`: output dimension of query and key layers in each attention head in each decoder layer. Usually `dim_k = dim_v = dim / num_heads`
- `dim_v`: output dimension of value layer in each attention head in each decoder layer
- `num_heads`: number of attention heads in each decoder layer
- `dim_ff`: hidden dimension of the feedforward sublayer in each decoder layer. Usually `dim_ff = 4 * dim`
- `num_layers`: number of decoder layers
- `dropout`: dropout probability for each sublayer in each decoder layer
- `σ`: activation function for the feedforward sublayer in each decoder layer (default: `gelu`)
- `no_encoder`: if `true`, ignore the encoder-decoder attention sublayer in each decoder layer. This is useful when no encoder output is provided.
- `linear_attention`: if `true`, use linear attention instead of dense scaled dot-product attention

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236.pdf)

# Example
```julia
decoder = Decoder(512, 64, 64, 8, 2048, 6);
src_seq_len = 10  # encoder sequence length
batch_size = 32
enc_out = randn(Float32, 512, src_seq_len, batch_size);  # encoder output
tgt_seq_len = 20
tgt = randn(Float32, 512, tgt_seq_len, batch_size);  # input to the decoder (already embedded and positional encoded)
dec_out = decoder(tgt, enc_out);   # size (512, tgt_seq_len, batch_size)
@assert size(dec_out) == (512, tgt_seq_len, batch_size)
# or if no encoder output is provided. `no_encoder` must be set to true
dec_out = decoder_layer(tgt)      # size (512, tgt_seq_len, batch_size)
@assert size(dec_out) == (512, tgt_seq_len, batch_size)
```
===
"""
struct Decoder
    layers::Vector{DecoderLayer}
end
Flux.@functor Decoder
function Decoder(dim::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_ff::Int, num_layers::Int; dropout=0.0, σ=gelu, no_encoder=false, linear_attention=false)
    layers = [DecoderLayer(dim, dim_k, dim_v, num_heads, dim_ff; dropout=dropout, σ=σ, no_encoder=no_encoder, linear_attention=linear_attention) for _ in 1:num_layers]
    return Decoder(layers)
end

"""
    (decoder::Decoder)(tgt, enc_out)

# Arguments
- `tgt`: target input to the decoder of shape (dim, seq_len_dec, batch_size)
- `enc_out`: output of the encoder of shape (dim, seq_len_enc, batch_size)

# Returns
- output of the last decoder layer of shape (dim, seq_len_dec, batch_size)

# Note: The `tgt` input to the decoder is assumed to be already embedded and positional encoded.
"""
function (decoder::Decoder)(tgt, enc_out)
    x = tgt
    for layer in decoder.layers
        x = layer(x, enc_out)
    end
    return x
end


"""
    (decoder::Decoder)(tgt)

    Ignores the encoder output and performs self-attention only i.e., the encoder-decoder attention sublayer is ignored in each decoder layer.

# Arguments
- `tgt`: target input to the decoder of shape (dim, seq_len_dec, batch_size)

# Returns
- output of the last decoder layer of shape (dim, seq_len_dec, batch_size)

# Note: The `tgt` input to the decoder is assumed to be already embedded and positional encoded.
"""
function (decoder::Decoder)(tgt)
    x = tgt
    for layer in decoder.layers
        x = layer(x)
    end
    return x
end


"""
    SinusoidalPositionalEncoder(dim::Int, max_seq_length::Int=10000)

Sinusoidal positional encoder. This is used to add positional information to the input embedding. The positional encoding is added to the embedding before the input is passed to the encoder or decoder.

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

# Arguments
- `dim`: dimensionality (1st dimension) of the positional encoding. This is the same as the dimensionality of the input to the positional encoder.
- `max_seq_length`: maximum sequence length that can be processed by the positional encoder (default: 10000)

# Example
```julia
spe = SinusoidalPositionalEncoder(512, 100);
x = randn(Float32, 512, 10);
y = spe(x)  # size (512, 10);
@assert size(y) == (512, 10)
```
"""
struct SinusoidalPositionalEncoder
    pe::AbstractArray{Float32, 2} # of shape (dim, max_seq_len)
end
Flux.gpu(spe::SinusoidalPositionalEncoder) = SinusoidalPositionalEncoder(gpu(spe.pe))
Flux.cpu(spe::SinusoidalPositionalEncoder) = SinusoidalPositionalEncoder(cpu(spe.pe))

function SinusoidalPositionalEncoder(dim::Int, max_seq_length::Int=10000)
    pe = zeros(Float32, (dim, max_seq_length))
    for pos in 1:max_seq_length
        for i in 1:dim
            logdenom = log(max_seq_length) * (2(i-1) / dim)
            denom = exp(logdenom)
            θ = pos / denom
            pe[i, pos] = i % 2 == 1 ? sin(θ) : cos(θ)
        end
    end
    return SinusoidalPositionalEncoder(pe)
end

function Base.show(io::IO, l::SinusoidalPositionalEncoder)
    print(io, "SinusoidalPositionalEncoder(", size(l.pe, 1), ")")
end

"""
    LearnedPositionalEncoder(dim::Int, max_seq_length::Int=10000, std=0.01f0)

Learned positional encoder. This is used to add positional information to the input embedding. The positional encoding is added to the embedding before the input is passed to the encoder or decoder. In this case, the positional encoding is a learnable parameter, initialized from a normal distribution. 

# Arguments
- `dim`: dimensionality (1st dimension) of the positional encoding. This is the same as the dimensionality of the input to the positional encoder.
- `max_seq_length`: maximum sequence length that can be processed by the positional encoder (default: 10000)
- `std`: standard deviation of the normal distribution used to initialize the positional encoding (default: 0.01f0)

# Example
```julia
lpe = LearnedPositionalEncoder(512, 100);
x = randn(Float32, 512, 10);
y = lpe(x);  # size (512, 10)
@assert size(y) == (512, 10)
@assert length(Flux.params(lpe)) == 1
```
"""
struct LearnedPositionalEncoder
    pe::AbstractArray{Float32, 2} # of shape (dim, max_seq_len)
end
Flux.@functor LearnedPositionalEncoder

function LearnedPositionalEncoder(dim::Int, max_seq_length::Int=10000, std=0.01f0)
    pe = Float32(std) * randn(Float32, (dim, max_seq_length))
    return LearnedPositionalEncoder(pe)
end


function (postional_encoder::Union{SinusoidalPositionalEncoder, LearnedPositionalEncoder})(x)
    @assert size(x, 1) == size(postional_encoder.pe, 1) "Dimensionality of the positional encoding and input to the positional encoder must be the same. Yours: $(size(x, 1)), expected: $(size(postional_encoder.pe, 1))"
    @assert size(x, 2) <= size(postional_encoder.pe, 2) "The sequence length of the input to the positional encoder cannot be greater than the maximum sequence length of the positional encoder. Yours: $(size(x, 2)), expected: not more than $(size(postional_encoder.pe, 2))"
    seq_len = size(x, 2)
    pe = @view postional_encoder.pe[:, 1:seq_len]
    return x .+ pe
end

function Base.show(io::IO, l::LearnedPositionalEncoder)
    print(io, "LearnedPositionalEncoder(", size(l.pe, 1), ")")
end
