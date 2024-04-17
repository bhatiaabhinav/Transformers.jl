using Flux
using CUDA
using LinearAlgebra
using DataStructures

function mybatchedtranspose(x)
    # return PermutedDimsArray(x, (2, 1, 3))
    batch_dims = size(x)[3:end]
    return permutedims(x, (2, 1, (1:ndims(x))[3:end]...))
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
In this implementation, `Q`, `K`, and `V` are batched matrices. The first dimension is the dimension of each vector, and the second dimension is the sequence length. After that, there can be any number of dimensions, which are treated as batch dimensions. Batch dimensions are optional.

Accordingly, `Q`, `K`, and `V` are of size `(d_k, seq_len_q, batch_size...)`, `(d_k, seq_len_k, batch_size...)`, and `(d_v, seq_len_k, batch_size...)`, respectively. The output is of size `(d_v, seq_len_q, batch_size...)`. Internally, the matrices are mutliplied in order opposite to the math notation. i.e., it is V * softmax(K^T * Q / sqrt(d_k)) since the original formula assumes row vectors instead of column vectors.

# Arguments
- `Q`: query of size `(d_k, seq_len_q, batch_size...)`
- `K`: key of size `(d_k, seq_len_k, batch_size...)`
- `V`: value of size `(d_v, seq_len_k, batch_size...)
- `masked`: whether to causal mask the attention scores

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
"""
function attention(Q, K, V, masked, dilation=1)
    d_k, seq_len_q = size(Q)[1:2]
    d_v, seq_len_k = size(V)[1:2]
    device = Q isa CUDA.CuArray ? gpu : cpu
    mask =  masked ? get_mask((seq_len_k, seq_len_q), device) : 0  # mask has -Inf at positions to be masked i.e., positions where key id > query id.
    batch_size_orig = size(Q)[3:end]
    batch_size = prod(batch_size_orig)
    Q = reshape(Q, d_k, seq_len_q, batch_size)
    K = reshape(K, d_k, seq_len_k, batch_size)
    V = reshape(V, d_v, seq_len_k, batch_size)
    if dilation > 1
        K = K[:, 1:dilation:end, :]
    end
    # println("K: ", size(K))
    # scores = mybatchedtranspose(K) ⊠ Q / Float32(sqrt(d_k)) # (seq_len_k, seq_len_q, batch_size)
    scores = batched_transpose(K) ⊠ Q / Float32(sqrt(d_k)) # (seq_len_k, seq_len_q, batch_size)
    # println("scores: ", size(scores))
    if dilation > 1
        scores = dilate_back(scores, (seq_len_k, seq_len_q, batch_size), dilation)
    end
    # println("scores final: ", size(scores))
    scores = scores .+ mask
    attn = softmax(scores, dims=1) # (seq_len_k, seq_len_q, batch_size)
    # Flux.Zygote.@ignore push!(attn_hisory, cpu(attn))
    ret_val = V ⊠ attn # (d_v, seq_len_q, batch_size)
    ret_val = reshape(ret_val, d_v, seq_len_q, batch_size_orig...)
    return ret_val
end

function dilate_back(A, output_size, step_size)
    B = Flux.Zygote.Buffer(A, output_size)
    B[:, :, :] = convert(typeof(A), fill(-Inf, size(B)))
    B[1:step_size:end, :, :] = A
    return copy(B)
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
Flux.@functor Attention

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
    MultiHeadAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int, masked::Bool)

Multi-head attention for Transformer. It is composed of multiple attention heads. The outputs of the attention heads are concatenated and passed through a linear layer.

# Arguments
- `dim_inp`: input dimension
- `dim_k`: output dimension of query and key layers.
- `dim_v`: output dimension of value layer. Usually `dim_v = dim_k = dim_inp / num_heads`
- `num_heads`: number of attention heads
- `dim_out`: output dimension of the (final) linear layer. Usually `dim_out = dim_inp`
- `masked`: whether to causal mask the attention scores in each attention head


# Examples
```julia
dim_inp, dim_k, dim_v = 512, 64, 64
num_heads = 8
dim_out = 512
mha = MultiHeadAttention(dim_inp, dim_k, dim_v, num_heads, dim_out, true);
seq_len_q = 10
seq_len_k = 20
batch_size = 32
q_input, k_input, v_input = randn(dim_inp, seq_len_q, batch_size), randn(dim_inp, seq_len_k, batch_size), randn(dim_inp, seq_len_k, batch_size); # (These inputs will be same for self attention. For encoder-decoder attention, key input and value input will be from encoder output, and query will be from decoder output.)
output = mha(q_input, k_input, v_input); # size (dim_out, seq_len_q, batch_size)
@assert size(output) == (dim_out, seq_len_q, batch_size)

```

"""
struct MultiHeadAttention
    heads::Vector{Attention}
    linear
    masked::Bool
end
Flux.@functor MultiHeadAttention

function MultiHeadAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int, masked::Bool)
    heads = [Attention(dim_inp, dim_k, dim_v, masked) for _ in 1:num_heads]
    linear = Dense(dim_v * num_heads, dim_out)
    return MultiHeadAttention(heads, linear, masked)
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
    # Each head outputs a tensor of size (dim_v, seq_len_q, batch_size). Concatenate the outputs of the attention heads along the first dimension to get a tensor of size (dim_v * num_heads, seq_len_q, batch_size).
    multihead_output = mapreduce(vcat, mha.heads) do head
        return head(q_inp, k_inp, v_inp) # (dim_v, seq_len_q, batch_size)
    end # (dim_v * num_heads, seq_len_q, batch_size)
    return mha.linear(multihead_output) # (dim_out, seq_len_q, batch_size)
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
Flux.@functor SelfAttention (qkv,)

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
    MultiHeadSelfAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int, maksed::Bool)

Equivalent to `MultiHeadAttention` but more efficient and expects the query input, key input, and value input to be the same. It is composed of a single linear layer for query, key, value and all heads, and applies self attention for each head in parallel. There is another linear layer as usual to map the output of the attention heads to the desired output dimension.

# Arguments
- `dim_inp`: input dimension
- `dim_k`: output dimension of query and key layers
- `dim_v`: output dimension of value layer. Usually `dim_v = dim_k = dim_inp / num_heads`
- `num_heads`: number of attention heads
- `dim_out`: output dimension of the (final) linear layer. Usually `dim_out = dim_inp`
- `masked`: whether to causal mask the attention scores in each head.

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
    cache
end

Flux.@functor MultiHeadSelfAttention (qkvh, linear)

function MultiHeadSelfAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int, maksed::Bool)
    qkvh = Dense(dim_inp, (dim_k * 2 + dim_v) * num_heads)
    linear = Dense(dim_v * num_heads, dim_out)
    return MultiHeadSelfAttention(qkvh, linear, dim_k, dim_v, num_heads, maksed, nothing)
end


"""
    (mhsa::MultiHeadSelfAttention)(x)

# Arguments
- `x`: input of size `(dim_inp, seq_len, batch_size)`
"""
function (mhsa::MultiHeadSelfAttention)(x)

    incremental = !haskey(ENV, "DISABLE_INCREMENTAL_ATTENTION") || ENV["DISABLE_INCREMENTAL_ATTENTION"] != "true"
    D, L, B = size(x, 1), size(x, 2), size(x)[3:end]
    if incremental && mhsa.cache === nothing
        incremental = false
    end
    if incremental
        prev_shape, q_old, k_old, v_old, cur_out_old = mhsa.cache
        prev_D, prev_L, prev_B = prev_shape[1], prev_shape[2], prev_shape[3:end]
        if prev_L != L - 1 || prev_D != D || prev_B != B
            incremental = false
        end
    end

    dim_k, dim_v, num_heads = mhsa.dim_k, mhsa.dim_v, mhsa.num_heads
    
    if incremental
        xnew = copy(selectdim(x, 2, L:L))  # (dim_inp, 1, batch_size)
        qkvh_new = mhsa.qkvh(xnew) # ((dim_k * 2 + dim_v) * num_heads, 1, batch_size)
        qkvh_new = reshape(qkvh_new, (dim_k * 2 + dim_v), num_heads, size(xnew)[2:end]...) # (dim_k * 2 + dim_v, num_heads, 1, batch_size)
        if ndims(x) == 2 # no batch dimension
            qkvh_new = permutedims(qkvh_new, (1, 3, 2)) # (dim_k * 2 + dim_v, 1, num_heads)
        else
            qkvh_new = permutedims(qkvh_new, (1, 3, 2, 4)) # (dim_k * 2 + dim_v, 1, num_heads, batch_size)
        end
        q_new, k_new, v_new = copy(selectdim(qkvh_new, 1, 1:dim_k)), copy(selectdim(qkvh_new, 1, dim_k+1:dim_k*2)), copy(selectdim(qkvh_new, 1, dim_k*2+1:dim_k*2+dim_v))
        q, k, v = cat(q_old, q_new, dims=2), cat(k_old, k_new, dims=2), cat(v_old, v_new, dims=2)
        if isa(x, CUDA.CuArray)
            CUDA.unsafe_free!(xnew)
            CUDA.unsafe_free!(qkvh_new)
            CUDA.unsafe_free!(k_new)
            CUDA.unsafe_free!(v_new)
            CUDA.unsafe_free!(q_old)
            CUDA.unsafe_free!(k_old)
            CUDA.unsafe_free!(v_old)
        end
    else
        qkvh = mhsa.qkvh(x) # ((dim_k * 2 + dim_v) * num_heads, seq_len, batch_size)
        qkvh = reshape(qkvh, (dim_k * 2 + dim_v), num_heads, size(x)[2:end]...) # (dim_k * 2 + dim_v, num_heads, seq_len, batch_size)
        if ndims(x) == 2 # no batch dimension
            qkvh = permutedims(qkvh, (1, 3, 2)) # (dim_k * 2 + dim_v, seq_len, num_heads)
        else
            qkvh = permutedims(qkvh, (1, 3, 2, 4)) # (dim_k * 2 + dim_v, seq_len, num_heads, batch_size)
        end
        q, k, v = copy(selectdim(qkvh, 1, 1:dim_k)), copy(selectdim(qkvh, 1, dim_k+1:dim_k*2)), copy(selectdim(qkvh, 1, dim_k*2+1:dim_k*2+dim_v))
    end


    if incremental
        multihead_output_new = attention(q_new, k, v, false) # (dim_v, 1, num_heads, batch_size)
        if ndims(x) == 2 # no batch dimension
            multihead_output_new = permutedims(multihead_output_new, (1, 3, 2)) # (dim_v, num_heads, 1)
        else
            multihead_output_new = permutedims(multihead_output_new, (1, 3, 2, 4)) # (dim_v, num_heads, 1, batch_size)
        end
        multihead_output_new = reshape(multihead_output_new, (dim_v * num_heads, size(multihead_output_new)[3:end]...)) # (dim_v * num_heads, 1, batch_size)
        cur_out_new = mhsa.linear(multihead_output_new) # (dim_out, 1, batch_size)
        cur_out = cat(cur_out_old, cur_out_new, dims=2) # (dim_out, seq_len, batch_size)
        if isa(x, CUDA.CuArray)
            CUDA.unsafe_free!(q_new)
            CUDA.unsafe_free!(multihead_output_new)
            CUDA.unsafe_free!(cur_out_new)
            CUDA.unsafe_free!(cur_out_old)
        end
    else
        multihead_output = attention(q, k, v, mhsa.masked) # (dim_v, seq_len, num_heads, batch_size)
        if ndims(x) == 2 # no batch dimension
            multihead_output = permutedims(multihead_output, (1, 3, 2)) # (dim_v, num_heads, seq_len)
        else
            multihead_output = permutedims(multihead_output, (1, 3, 2, 4)) # (dim_v, num_heads, seq_len, batch_size)
        end
        multihead_output = reshape(multihead_output, (dim_v * num_heads, size(multihead_output)[3:end]...)) # (dim_v * num_heads, seq_len, batch_size)
        cur_out = mhsa.linear(multihead_output) # (dim_out, seq_len, batch_size)
    end
    mhsa.cache = (size(x), q, k, v, cur_out)
    return cur_out
end




"""
    MultiHeadLinearSelfAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int, maksed::Bool)

A linear approximation to `MultiHeadSelfAttention`. O(n) in time and memory complexity. O(1) inference complexity (in causal mode).

# Arguments
- `dim_inp`: input dimension
- `dim_k`: output dimension of query and key layers
- `dim_v`: output dimension of value layer. Usually `dim_v = dim_k = dim_inp / num_heads`
- `num_heads`: number of attention heads
- `dim_out`: output dimension of the (final) linear layer. Usually `dim_out = dim_inp`
- `masked`: whether to causal mask the attention scores in each head.

# References
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236.pdf)

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
mutable struct MultiHeadLinearSelfAttention
    qkvh # a combined linear layer for query, key, value and heads
    linear
    dim_k::Int
    dim_v::Int
    num_heads::Int
    masked::Bool
    cache
end

Flux.@functor MultiHeadLinearSelfAttention (qkvh, linear)

function MultiHeadLinearSelfAttention(dim_inp::Int, dim_k::Int, dim_v::Int, num_heads::Int, dim_out::Int, maksed::Bool)
    qkvh = Dense(dim_inp, (dim_k * 2 + dim_v) * num_heads)
    linear = Dense(dim_v * num_heads, dim_out)
    return MultiHeadLinearSelfAttention(qkvh, linear, dim_k, dim_v, num_heads, maksed, nothing)
end


"""
    (mhsa::MultiHeadLinearSelfAttention)(x)

# Arguments
- `x`: input of size `(dim_inp, seq_len, batch_size)`
"""
function (mhsa::MultiHeadLinearSelfAttention)(x)

    incremental = !haskey(ENV, "DISABLE_INCREMENTAL_ATTENTION") || ENV["DISABLE_INCREMENTAL_ATTENTION"] != "true"
    D, L, B = size(x, 1), size(x, 2), size(x)[3:end]
    if incremental && mhsa.cache === nothing
        incremental = false
    end
    linear_attn_cache = nothing
    if incremental
        prev_size, q_old, k_old, v_old, cur_out_old, linear_attn_cache = mhsa.cache
        prev_D, prev_L, prev_B = prev_size[1], prev_size[2], prev_size[3:end]
        if !mhsa.masked || prev_L != L - 1 || prev_D != D || prev_B != B
            incremental = false
        end
    end

    dim_k, dim_v, num_heads = mhsa.dim_k, mhsa.dim_v, mhsa.num_heads
    
    if incremental
        xnew = copy(selectdim(x, 2, L:L))  # (dim_inp, 1, batch_size)
        qkvh_new = mhsa.qkvh(xnew) # ((dim_k * 2 + dim_v) * num_heads, 1, batch_size)
        qkvh_new = reshape(qkvh_new, (dim_k * 2 + dim_v), num_heads, size(xnew)[2:end]...) # (dim_k * 2 + dim_v, num_heads, 1, batch_size)
        if ndims(x) == 2 # no batch dimension
            qkvh_new = permutedims(qkvh_new, (1, 3, 2)) # (dim_k * 2 + dim_v, 1, num_heads)
        else
            qkvh_new = permutedims(qkvh_new, (1, 3, 2, 4)) # (dim_k * 2 + dim_v, 1, num_heads, batch_size)
        end
        q_new, k_new, v_new = copy(selectdim(qkvh_new, 1, 1:dim_k)), copy(selectdim(qkvh_new, 1, dim_k+1:dim_k*2)), copy(selectdim(qkvh_new, 1, dim_k*2+1:dim_k*2+dim_v))
        q, k, v = cat(q_old, q_new, dims=2), cat(k_old, k_new, dims=2), cat(v_old, v_new, dims=2)
        if isa(x, CUDA.CuArray)
            CUDA.unsafe_free!(xnew)
            CUDA.unsafe_free!(qkvh_new)
            CUDA.unsafe_free!(q_old)
            CUDA.unsafe_free!(k_old)
            CUDA.unsafe_free!(v_old)
        end
    else
        qkvh = mhsa.qkvh(x) # ((dim_k * 2 + dim_v) * num_heads, seq_len, batch_size)
        qkvh = reshape(qkvh, (dim_k * 2 + dim_v), num_heads, size(x)[2:end]...) # (dim_k * 2 + dim_v, num_heads, seq_len, batch_size)
        if ndims(x) == 2 # no batch dimension
            qkvh = permutedims(qkvh, (1, 3, 2)) # (dim_k * 2 + dim_v, seq_len, num_heads)
        else
            qkvh = permutedims(qkvh, (1, 3, 2, 4)) # (dim_k * 2 + dim_v, seq_len, num_heads, batch_size)
        end
        q, k, v = copy(selectdim(qkvh, 1, 1:dim_k)), copy(selectdim(qkvh, 1, dim_k+1:dim_k*2)), copy(selectdim(qkvh, 1, dim_k*2+1:dim_k*2+dim_v))
    end


    if incremental
        if L > 1 && linear_attn_cache === nothing
            println("this is bad")
            @warn "Incremental input detected but somehow no cache available for linear attention. Disabling incremental mode for this part of the computation."
            incremental = false
        end
    end

    if incremental
        linear_attn_cache_old = linear_attn_cache
        multihead_output_new, linear_attn_cache = linear_attention_causal_incremental!(q_new, k_new, v_new, linear_attn_cache_old) # (dim_v, 1, num_heads, batch_size)
        if ndims(x) == 2 # no batch dimension
            multihead_output_new = permutedims(multihead_output_new, (1, 3, 2)) # (dim_v, num_heads, 1)
        else
            multihead_output_new = permutedims(multihead_output_new, (1, 3, 2, 4)) # (dim_v, num_heads, 1, batch_size)
        end
        multihead_output_new = reshape(multihead_output_new, (dim_v * num_heads, size(multihead_output_new)[3:end]...)) # (dim_v * num_heads, 1, batch_size)
        cur_out_new = mhsa.linear(multihead_output_new) # (dim_out, 1, batch_size)
        cur_out = cat(cur_out_old, cur_out_new, dims=2) # (dim_out, seq_len, batch_size)
        if isa(x, CUDA.CuArray)
            CUDA.unsafe_free!(q_new)
            CUDA.unsafe_free!(k_new)
            CUDA.unsafe_free!(v_new)
            CUDA.unsafe_free!(multihead_output_new)
            CUDA.unsafe_free!(cur_out_new)
            CUDA.unsafe_free!(cur_out_old)
            CUDA.unsafe_free!(linear_attn_cache_old[1])
            CUDA.unsafe_free!(linear_attn_cache_old[2])
        end
    else
        if mhsa.masked
            multihead_output, linear_attn_cache = linear_attention(q, k, v, true; return_cache=true) # (dim_v, seq_len, num_heads, batch_size)
        else
            multihead_output = linear_attention(q, k, v, false) # (dim_v, seq_len, num_heads, batch_size)
        end
        if ndims(x) == 2 # no batch dimension
            multihead_output = permutedims(multihead_output, (1, 3, 2)) # (dim_v, num_heads, seq_len)
        else
            multihead_output = permutedims(multihead_output, (1, 3, 2, 4)) # (dim_v, num_heads, seq_len, batch_size)
        end
        multihead_output = reshape(multihead_output, (dim_v * num_heads, size(multihead_output)[3:end]...)) # (dim_v * num_heads, seq_len, batch_size)
        cur_out = mhsa.linear(multihead_output) # (dim_out, seq_len, batch_size)
    end
    mhsa.cache = (size(x), q, k, v, cur_out, linear_attn_cache)
    return cur_out
end