using Flux
using CUDA
using LinearAlgebra
using DataStructures

function _transpose(x::AbstractArray{T, N})::AbstractArray{T, N} where {T, N}
    return permutedims(x, (2, 1, (1:ndims(x))[3:end]...))
end
function _transpose(x::AbstractVecOrMat{T})::AbstractVecOrMat{T} where {T}
    return x'
end

cached_masks = Dict{Any, Any}()


function get_mask(shape, device)
    return Flux.Zygote.ignore() do
        if haskey(cached_masks, (shape, device))
            return cached_masks[(shape, device)]
        else
            mask = device(-1 ./ triu(fill(-Inf32, shape)))
            cached_masks[(shape, device)] = mask
            return mask
        end
    end
end

attn_hisory = CircularBuffer{Any}(10)

"""
    attention(Q, K, V)

Attention mechanism for Transformer. It is composed of three linear layers for query, key, and value. It is defined as

```math
Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V
```

where `Q`, `K`, and `V` are query, key, and value, respectively. `d_k` is the dimension of `K` and `Q`.
In this implementation, `Q`, `K`, and `V` are batched matrices. The first dimension is the dimension of each vector, and the second dimension is the sequence length, and the third dimension onwards are batch dimensions.

Accordingly, `Q`, `K`, and `V` are of size `(d_k, seq_len_q, batch_size...)`, `(d_k, seq_len_k, batch_size...)`, and `(d_v, seq_len_k, batch_size...)`, respectively. The output is of size `(d_v, seq_len_q, batch_size...)`. Internally, the matrices are mutliplied in order opposite to the math notation. i.e., it is V * softmax(Kᵀ * Q / sqrt(d_k)) since the original formula in the paper assumed row vectors instead of column vectors. (Column vectors are more efficient in Julia and also respect mathematical conventions).

# Arguments
- `Q`: query of size `(d_k, seq_len_q, batch_size)`
- `K`: key of size `(d_k, seq_len_k, batch_size)`
- `V`: value of size `(d_v, seq_len_k, batch_size)
- `masked`: whether to causal mask the attention scores

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
"""
function attention(Q, K, V, masked)
    d_k, seq_len_q = size(Q)[1:2]
    d_v, seq_len_k = size(V)[1:2]
    device = Q isa CUDA.CuArray ? gpu : cpu
    scores = _transpose(K) ⊠ Q / Float32(sqrt(d_k))        # (seq_len_k, seq_len_q, batch_size)
    if masked
        @assert seq_len_q == seq_len_k "Masked attention requires the query and key sequences to be of the same length"
        mask = get_mask((seq_len_k, seq_len_q), device)
        scores = scores .+ mask
    end
    attn = softmax(scores, dims=1)              # (seq_len_k, seq_len_q, batch_size)
    ret_val = V ⊠ attn                          # (d_v, seq_len_q, batch_size...)
    return ret_val
end


mutable struct CasualAttentionWithKVCaching
    cache # K, V, output
end
Flux.@layer :ignore CasualAttentionWithKVCaching trainable=()
Base.show(::IO, ::CasualAttentionWithKVCaching) = print("CasualAttentionIncremental()")
CasualAttentionWithKVCaching() = CasualAttentionWithKVCaching(nothing)
function reset_kv_cache!(ca::CasualAttentionWithKVCaching)
    free_cache_memory!(ca)
end
function free_cache_memory!(cai::CasualAttentionWithKVCaching)
    if !isnothing(cai.cache)
        K, V = cai.cache
        Flux.Zygote.@ignore if isa(K, CUDA.CuArray); CUDA.unsafe_free!(K); end
        Flux.Zygote.@ignore if isa(V, CUDA.CuArray); CUDA.unsafe_free!(V); end
        cai.cache = nothing
    end
end

function (cai::CasualAttentionWithKVCaching)(Q_new, K_new, V_new, masked)
    @assert masked "Incremental attention caching is supported only for causal (masked) attention"
    if isnothing(cai.cache)
        Q, K, V = Q_new, K_new, V_new  # assuming that the first call is not incremental and these can be full length
        ret = attention(Q, K, V, true)
        cai.cache = Flux.Zygote.@ignore (K, V)
        return ret
    else
        @assert size(Q_new)[2] == 1 "Only one query at a time is allowed for incremental attention. Call Transformers.reset_kv_cache! to reset the cache to allow a sequence of queries."
        K = Flux.Zygote.@ignore cat(cai.cache[1], K_new, dims=2)
        V = Flux.Zygote.@ignore cat(cai.cache[2], V_new, dims=2)
        free_cache_memory!(cai)
        cai.cache = Flux.Zygote.@ignore (K, V)
        return attention(Q_new, K, V, false)
    end
end




"""
    Attention(dim_inp::Int, dim_k::Int, dim_v::Int, masked::Bool)

Attention head for a Transformer. It is composed of three linear layers for query, key, and value.

# Arguments
- `dim_inp`: input dimension
- `dim_k`: output dimension of query and key layers
- `dim_v`: output dimension of value layer
- `masked`: whether to causal mask the attention scores

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

# Examples
```julia
dim_inp, dim_k, dim_v = 512, 64, 64;
attnhead = Attention(dim_inp, dim_k, dim_v, true);
seq_len_q, seq_len_k, batch_size = 10, 20, 32;
q_input, k_input, v_input = randn(Float32, dim_inp, seq_len_q, batch_size), randn(Float32, dim_inp, seq_len_k, batch_size), randn(Float32, dim_inp, seq_len_k, batch_size); # (Usually, these inputs will be same for self attention. For encoder-decoder attention, key input and value input will be from encoder output, and query will be from decoder output.)
output = attnhead(q_input, k_input, v_input); # size (dim_v, seq_len_q, batch_size)
@assert size(output) == (dim_v, seq_len_q, batch_size)
```

"""
struct Attention
    q_layer::Dense
    k_layer::Dense
    v_layer::Dense
    masked::Bool
end
Flux.@layer Attention

function Attention(dim_inp::Int, dim_k::Int, dim_v::Int, masked::Bool)
    q_layer = Dense(dim_inp, dim_k)
    k_layer = Dense(dim_inp, dim_k)
    v_layer = Dense(dim_inp, dim_v)
    return Attention(q_layer, k_layer, v_layer, masked)
end


"""
    (head::Attention)(q_inp, k_inp, v_inp)

# Arguments
- `q_inp`: query input of size `(dim_inp, seq_len_q, batch_size)`
- `k_inp`: key input of size `(dim_inp, seq_len_k, batch_size)`
- `v_inp`: value input of size `(dim_inp, seq_len_k, batch_size)`

# Returns
- `output`: output of size `(dim_v, seq_len_q, batch_size)`
"""
function (head::Attention)(q_inp, k_inp, v_inp)
    split
    q = head.q_layer(q_inp)  # (dim_k, seq_len_q, batch_size)
    k = head.k_layer(k_inp)  # (dim_k, seq_len_k, batch_size)
    v = head.v_layer(v_inp)  # (dim_v, seq_len_k, batch_size)
    return attention(q, k, v, head.masked)  # (dim_v, seq_len_q, batch_size)
end


"""
    MultiHeadAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int)

Multi-head attention for Transformer. It is composed of multiple attention heads. The outputs of the attention heads are concatenated and passed through a linear layer.

# Arguments
- `dim_inp`: input dimension
- `dim_k`: output dimension of query and key layers.
- `dim_v`: output dimension of value layer. Usually `dim_v = dim_k = dim_inp / num_heads`
- `num_heads`: number of attention heads
- `dim_out`: output dimension of the (final) linear layer. Usually `dim_out = dim_inp`


# Examples
```julia
dim_inp, dim_k, dim_v = 512, 64, 64
num_heads = 8
dim_out = 512
mha = MultiHeadAttention(dim_inp, dim_k, dim_v, num_heads, dim_out);
seq_len_q = 10
seq_len_k = 20
batch_size = 32
q_input, k_input, v_input = randn(dim_inp, seq_len_q, batch_size), randn(dim_inp, seq_len_k, batch_size), randn(dim_inp, seq_len_k, batch_size); # (These inputs will be same for self attention. For encoder-decoder attention, key input and value input will be from encoder output, and query will be from decoder output.)
output = mha(q_input, k_input, v_input); # size (dim_out, seq_len_q, batch_size)
@assert size(output) == (dim_out, seq_len_q, batch_size)

```

"""
struct MultiHeadAttention
    qh  # a combined linear layer for query and heads
    kh  # a combined linear layer for key and heads
    vh  # a combined linear layer for value and heads
    linear
    dim_k::Int
    dim_v::Int
    num_heads::Int
end
Flux.@layer MultiHeadAttention

function MultiHeadAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int)
    qh = Dense(dim_inp, dim_k * num_heads)
    vh = Dense(dim_inp, dim_v * num_heads)
    kh = Dense(dim_inp, dim_k * num_heads)
    linear = Dense(dim_v * num_heads, dim_out)
    return MultiHeadAttention(qh, kh, vh, linear, dim_k, dim_v, num_heads)
end


"""
    (mha::MultiHeadAttention)(q_inp, k_inp, v_inp)

# Arguments
- `q_inp`: query input of size `(dim_inp, seq_len_q, batch_size)`
- `k_inp`: key input of size `(dim_inp, seq_len_k, batch_size)`
- `v_inp`: value input of size `(dim_inp, seq_len_k, batch_size)`

# Returns
- `output`: output of size `(dim_out, seq_len_q, batch_size)`
"""
function (mha::MultiHeadAttention)(q_inp, k_inp, v_inp)
    dim_k, dim_v, num_heads = mha.dim_k, mha.dim_v, mha.num_heads
    seq_len_q = size(q_inp)[2]
    seq_len_k = size(k_inp)[2]
    seq_len_v = size(v_inp)[2]
    @assert seq_len_k == seq_len_v "Key and value inputs should have the same sequence length"
    q = mha.qh(q_inp)  # (dim_k * num_heads, seq_len_q, batch_size)
    k = mha.kh(k_inp)  # (dim_k * num_heads, seq_len_k, batch_size)
    v = mha.vh(v_inp)  # (dim_v * num_heads, seq_len_k, batch_size)
    q = reshape(q, (dim_k, num_heads, size(q)[2:end]...))  # (dim_k, num_heads, seq_len_q, batch_size)
    k = reshape(k, (dim_k, num_heads, size(k)[2:end]...))  # (dim_k, num_heads, seq_len_k, batch_size)
    v = reshape(v, (dim_v, num_heads, size(v)[2:end]...))  # (dim_v, num_heads, seq_len_k, batch_size)
    if ndims(q_inp) == 2
        q = permutedims(q, (1, 3, 2))  # (dim_k, seq_len_q, num_heads)
        k = permutedims(k, (1, 3, 2))  # (dim_k, seq_len_k, num_heads)
        v = permutedims(v, (1, 3, 2))  # (dim_v, seq_len_k, num_heads)
    else
        q = permutedims(q, (1, 3, 2, 4))  # (dim_k, seq_len_q, num_heads, batch_size)
        k = permutedims(k, (1, 3, 2, 4))  # (dim_k, seq_len_k, num_heads, batch_size)
        v = permutedims(v, (1, 3, 2, 4))  # (dim_v, seq_len_k, num_heads, batch_size)
    end
    multihead_output = attention(q, k, v, false)  # (dim_v, seq_len_q, num_heads, batch_size)
    if ndims(q_inp) == 2
        multihead_output = permutedims(multihead_output, (1, 3, 2))  # (dim_v, num_heads, seq_len_q)
    else
        multihead_output = permutedims(multihead_output, (1, 3, 2, 4))  # (dim_v, num_heads, seq_len_q, batch_size)
    end
    multihead_output = reshape(multihead_output, (dim_v * num_heads, size(multihead_output)[3:end]...))  # (dim_v * num_heads, seq_len_q, batch_size)
    out = mha.linear(multihead_output)  # (dim_out, seq_len_q, batch_size)
    return out
end





"""
    SelfAttention(dim_inp::Int, dim_k::Int, dim_v::Int, masked::Bool)

Equivalent to `Attention` but more efficient and expects the query input, key input, and value input to be the same. It is composed of a single linear layer for query, key, and value so that the computation can be parallelized by having a single matrix multiplication.

# Arguments
- `dim_inp`: input dimension
- `dim_k`: output dimension of query and key layers
- `dim_v`: output dimension of value layer
- `masked`: whether to causal mask the attention scores.

# Examples
```julia
dim_inp, dim_k, dim_v = 512, 64, 64
sa = SelfAttention(dim_inp, dim_k, dim_v, true);
seq_len = 20
batch_size = 32
input = randn(dim_inp, seq_len, batch_size);
output = sa(input) # size (dim_v, seq_len, batch_size);
@assert size(output) == (dim_v, seq_len, batch_size)
```
===
"""
struct SelfAttention
    qkv::Dense # a combined linear layer for query, key, and value
    dim_k::Int
    dim_v::Int
    masked::Bool
end
Flux.@layer SelfAttention trainable=(qkv,)

function SelfAttention(dim_inp::Int, dim_k::Int, dim_v::Int, masked::Bool)
    qkv = Dense(dim_inp, dim_k * 2 + dim_v)
    return SelfAttention(qkv, dim_k, dim_v, masked)
end

"""
    (self::SelfAttention)(x)

# Arguments
- `x`: input of size `(dim_inp, seq_len, batch_size)`
"""
function (self::SelfAttention)(x)
    qkv = self.qkv(x) # (dim_k * 2 + dim_v, seq_len, batch_size)
    dim_k, dim_v = self.dim_k, self.dim_v
    q, k, v = copy(selectdim(qkv, 1, 1:dim_k)), copy(selectdim(qkv, 1, dim_k+1:dim_k*2)), copy(selectdim(qkv, 1, dim_k*2+1:dim_k*2+dim_v))
    return attention(q, k, v, self.masked) # (dim_v, seq_len, batch_size)
end



"""
    MultiHeadSelfAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int, maksed::Bool; incremental_inference_mode=false)

Equivalent to `MultiHeadAttention` but more efficient and expects the query input, key input, and value input to be the same. It is composed of a single linear layer for query, key, value and all heads, and applies self attention for each head in parallel. There is another linear layer as usual to map the output of the attention heads to the desired output dimension.

# Arguments
- `dim_inp`: input dimension
- `dim_k`: output dimension of query and key layers
- `dim_v`: output dimension of value layer. Usually `dim_v = dim_k = dim_inp / num_heads`
- `num_heads`: number of attention heads
- `dim_out`: output dimension of the (final) linear layer. Usually `dim_out = dim_inp`
- `masked`: whether to causal mask the attention scores in each head.
- `incremental_inference_mode`: whether to enable incremental caching for causal attention. This is useful for auto-regressive or incremental decoding for faster/linear inference at each time step. When enabled, only one input should be passed to the model at a time (the previous KVs are already cached) and you should call `Flux.reset!` to reset the cache to allow a sequence of inputs at once (as usual e.g., in training).

# Examples
```julia
dim_inp, dim_k, dim_v = 512, 64, 64
num_heads = 8
mhsa = MultiHeadSelfAttention(dim_inp, dim_k, dim_v, num_heads, dim_inp, true);
seq_len = 20
batch_size = 32
input = randn(dim_inp, seq_len, batch_size);
output = mhsa(input); # size (dim_out, seq_len, batch_size)
@assert size(output) == (dim_inp, seq_len, batch_size)
```
===
"""
mutable struct MultiHeadSelfAttention
    qkvh # a combined linear layer for query, key, value and heads
    linear
    dim_k::Int
    dim_v::Int
    num_heads::Int
    masked::Bool
    attn_fn
end

Flux.@layer MultiHeadSelfAttention trainable=(qkvh, linear)
function reset_kv_cache!(mhsa::MultiHeadSelfAttention)
    isa(mhsa.attn_fn, CasualAttentionWithKVCaching) ? reset_kv_cache!(mhsa.attn_fn) : nothing
    return nothing
end

function MultiHeadSelfAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int, masked::Bool; incremental_inference_mode=false)
    qkvh = Dense(dim_inp, (dim_k * 2 + dim_v) * num_heads)
    linear = Dense(dim_v * num_heads, dim_out)
    attn_fn = incremental_inference_mode ? CasualAttentionWithKVCaching() : attention
    return MultiHeadSelfAttention(qkvh, linear, dim_k, dim_v, num_heads, masked, attn_fn)
end


"""
    (mhsa::MultiHeadSelfAttention)(x)

# Arguments
- `x`: input of size `(dim_inp, seq_len, batch_size)`
"""
function (mhsa::MultiHeadSelfAttention)(x)
    dim_k, dim_v, num_heads = mhsa.dim_k, mhsa.dim_v, mhsa.num_heads
    qkvh = mhsa.qkvh(x) # ((dim_k * 2 + dim_v) * num_heads, seq_len, batch_size)
    qkvh = reshape(qkvh, (dim_k * 2 + dim_v), num_heads, size(x)[2:end]...) # (dim_k * 2 + dim_v, num_heads, seq_len, batch_size)
    if ndims(x) == 2 # no batch dimension
        qkvh = permutedims(qkvh, (1, 3, 2)) # (dim_k * 2 + dim_v, seq_len, num_heads)
    else
        qkvh = permutedims(qkvh, (1, 3, 2, 4)) # (dim_k * 2 + dim_v, seq_len, num_heads, batch_size)
    end
    q, k, v = copy(selectdim(qkvh, 1, 1:dim_k)), copy(selectdim(qkvh, 1, dim_k+1:dim_k*2)), copy(selectdim(qkvh, 1, dim_k*2+1:dim_k*2+dim_v))
    multihead_output = mhsa.attn_fn(q, k, v, mhsa.masked) # (dim_v, seq_len, num_heads, batch_size)
    if ndims(x) == 2 # no batch dimension
        multihead_output = permutedims(multihead_output, (1, 3, 2)) # (dim_v, num_heads, seq_len)
    else
        multihead_output = permutedims(multihead_output, (1, 3, 2, 4)) # (dim_v, num_heads, seq_len, batch_size)
    end
    multihead_output = reshape(multihead_output, (dim_v * num_heads, size(multihead_output)[3:end]...)) # (dim_v * num_heads, seq_len, batch_size)
    out = mhsa.linear(multihead_output) # (dim_out, seq_len, batch_size)
    return out
end
