using Flux
using CUDA

Base.adjoint(x::AbstractArray{T, 3}) where T = batched_transpose(x)
Base.adjoint(x::AbstractArray{T, 4}) where T = mybatchedtranspose(x)
const A3D{T} = AbstractArray{T, 3}  # 3D array type alias for readability
const AnD{T, N} = AbstractArray{T, N}  # N-D array type alias for readability

function ϕ(x::AbstractArray{T, N}) where {T, N}
    elu(x) .+ T(1)
end

function linear_attention(Q::A3D{T}, K::A3D{T}, V::A3D{T}, causal::Bool; return_cache=false) where {T<:AbstractFloat}
    if causal
        attn_output, cache = linear_attention_causal(Q, K, V)
        return return_cache ? (attn_output, cache) : attn_output
    else
        return linear_attention_non_causal(Q, K, V)
    end
end

function linear_attention(Q::AnD{T, N}, K::AnD{T, N}, V::AnD{T, N}, causal::Bool; return_cache=false) where {T<:AbstractFloat, N}
    batch_shape = size(Q)[3:end]
    dim_k, seq_len_q = size(Q, 1), size(Q, 2)
    dim_v, seq_len_k = size(V, 1), size(V, 2)
    _Q = reshape(Q, dim_k, seq_len_q, :)
    _K = reshape(K, dim_k, seq_len_k, :)
    _V = reshape(V, dim_v, seq_len_k, :)
    if causal
        attention_output, cache = linear_attention_causal(_Q, _K, _V)
        _S, _Z = cache
        S = reshape(_S, size(_S, 1), size(_S, 2), batch_shape...)
        Z = reshape(_Z, size(_Z, 1), size(_Z, 2), batch_shape...)
        attention_output = reshape(attention_output, dim_v, seq_len_q, batch_shape...)
        return return_cache ? (attention_output, (S, Z)) : attention_output
    else
        return linear_attention_non_causal(_Q, _K, _V)
    end
end

function linear_attention_non_causal(Q::A3D{T}, K::A3D{T}, V::A3D{T})::A3D{T} where T
    ϕQ, ϕK  = ϕ(Q), ϕ(K)
    S = V ⊠ ϕK'                                     # (dim_v, dim_p, :)
    Z = sum(ϕK, dims=2)                             # (dim_p, 1, :)
    numerator = S ⊠ ϕQ                              # (dim_v, seq_len_q, :)
    denominator = Z' ⊠ ϕQ                           # (1, seq_len_q, :)
    attention_output = numerator ./ denominator     # (dim_v, seq_len_q, :)
    return attention_output
end

function linear_attention_causal(Q::A3D{T}, K::A3D{T}, V::A3D{T}) where T
    ϕQ, ϕK  = ϕ(Q), ϕ(K)                            # (dim_p, seq_len, :), (dim_p, seq_len, :)
    numerator, S = linear_attention_causal_numerator(ϕQ, ϕK, V)
    denominator, Z = linear_attention_causal_denominator(ϕQ, ϕK)
    attention_output = numerator ./ denominator  # (dim_v, seq_len, :)
    return attention_output, (S, Z)
end

function linear_attention_causal_numerator(ϕQ::A3D{T}, ϕK::A3D{T}, V::A3D{T}) where T
    dim_p, seq_len, batch_size = size(ϕQ)
    dim_v = size(V, 1)
    zerosfn = ϕQ isa CuArray ? CUDA.zeros : zeros
    S = zerosfn(T, dim_v, dim_p, batch_size)
    VᵢϕKᵢᵀ = zerosfn(T, dim_v, dim_p, batch_size)
    numerator = zerosfn(T, dim_v, seq_len, batch_size)
    @views for i in 1:seq_len
        ϕQᵢ, ϕKᵢ, Vᵢ  = ϕQ[:, i:i, :], ϕK[:, i:i, :], V[:, i:i, :]
        batched_mul!(VᵢϕKᵢᵀ, Vᵢ, ϕKᵢ')
        S .+= VᵢϕKᵢᵀ
        batched_mul!(numerator[:, i:i, :], S, ϕQᵢ)
    end
    return numerator, S
end

function linear_attention_causal_denominator(ϕQ::A3D{T}, ϕK::A3D{T}) where T
    dim_p, seq_len, batch_size = size(ϕQ)
    zerosfn = ϕQ isa CuArray ? CUDA.zeros : zeros
    Z = zerosfn(T, dim_p, 1, batch_size)
    denominator = zerosfn(T, 1, seq_len, batch_size)
    @views for i in 1:seq_len
        ϕQᵢ, ϕKᵢ = ϕQ[:, i:i, :], ϕK[:, i:i, :]
        Z .+= ϕKᵢ                                               # (dim_p, 1, batch_size)
        batched_mul!(denominator[:, i:i, :], Z', ϕQᵢ)         # (1, 1, batch_size)
    end
    return denominator, Z
end


function linear_attention_causal_incremental!(Qᵢ::AnD{T, N}, Kᵢ::AnD{T, N}, Vᵢ::AnD{T, N}, cache=nothing) where {T<:AbstractFloat, N}
    ϕQᵢ, ϕKᵢ = ϕ(Qᵢ), ϕ(Kᵢ)
    dim_p, batch_shape = size(ϕQᵢ, 1), size(ϕQᵢ)[3:end]
    dim_v = size(Vᵢ, 1)
    zerosfn = ϕQᵢ isa CuArray ? CUDA.zeros : zeros
    if isnothing(cache)
        Sᵢ₋₁ = zerosfn(T, dim_v, dim_p, batch_shape...)
        Zᵢ₋₁ = zerosfn(T, dim_p, 1, batch_shape...)
    else
        Sᵢ₋₁, Zᵢ₋₁ = cache
    end
    Sᵢ = Sᵢ₋₁ + Vᵢ ⊠ ϕKᵢ'  # (dim_v, dim_p, batch_shape...)
    Zᵢ = Zᵢ₋₁ + ϕKᵢ
    numeratorᵢ = Sᵢ ⊠ ϕQᵢ                           # (dim_v, 1, batch_shape...)
    denominatorᵢ = Zᵢ' ⊠ ϕQᵢ     # (1, 1, batch_shape...)
    attention_outputᵢ = numeratorᵢ ./ denominatorᵢ  # (dim_v, 1, batch_shape...)
    return attention_outputᵢ, (Sᵢ, Zᵢ)
end






# ---------------------------------------------- Adjoints ----------------------------------------------

Flux.Zygote.@adjoint function linear_attention_causal_numerator(ϕQ::A3D{T}, ϕK::A3D{T}, V::A3D{T}) where T
    output = linear_attention_causal_numerator(ϕQ, ϕK, V)
    function pullback(Δ)  # Δ is of shape (dim_v, seq_len, :)
        Δ = Δ[1]  # we are interested only in gradients w.r.t. the numerator
        dim_v, seq_len, batch_size = size(Δ)
        dim_p = size(ϕK, 1)
        zerosfn = ϕQ isa CuArray ? CUDA.zeros : zeros
        Δ_ϕQ = zerosfn(T, dim_p, seq_len, batch_size)
        Δ_ϕK = zerosfn(T, dim_p, seq_len, batch_size)
        Δ_V = zerosfn(T, dim_v, seq_len, batch_size)
        S = zerosfn(T, dim_v, dim_p, batch_size)
        VᵢϕKᵢᵀ = zerosfn(T, dim_v, dim_p, batch_size)
        @views for i in 1:seq_len
            ϕKᵢ, Vᵢ, Gᵢ  = ϕK[:, i:i, :], V[:, i:i, :], Δ[:, i:i, :]
            batched_mul!(VᵢϕKᵢᵀ, Vᵢ, ϕKᵢ')
            S .+= VᵢϕKᵢᵀ
            batched_mul!(Δ_ϕQ[:, i:i, :], S', Gᵢ)
        end
        fill!(S, T(0))
        GᵢϕQᵢᵀ = zerosfn(T, dim_v, dim_p, batch_size)  # as a buffer
        @views for i in reverse(1:seq_len)
            ϕQᵢ, ϕKᵢ, Vᵢ, Gᵢ = ϕQ[:, i:i, :], ϕK[:, i:i, :], V[:, i:i, :], Δ[:, i:i, :]
            batched_mul!(GᵢϕQᵢᵀ, Gᵢ, ϕQᵢ')
            S .+= GᵢϕQᵢᵀ
            batched_mul!(Δ_V[:, i:i, :], S, ϕKᵢ)
            batched_mul!(Δ_ϕK[:, i:i, :], S', Vᵢ)
        end
        (Δ_ϕQ, Δ_ϕK, Δ_V)
    end
    return output, pullback
end

Flux.Zygote.@adjoint function linear_attention_causal_denominator(ϕQ::A3D{T}, ϕK::A3D{T}) where T
    output = linear_attention_causal_denominator(ϕQ, ϕK)
    function pullback(Δ)  # Δ is of shape (1, seq_len_q, :)
        Δ = Δ[1]  # we are interested only in gradients w.r.t. the denominator
        dim_p, seq_len, batch_size = size(ϕQ)
        zerosfn = ϕQ isa CuArray ? CUDA.zeros : zeros
        Δ_ϕQ = zerosfn(T, dim_p, seq_len, batch_size)
        Δ_ϕK = zerosfn(T, dim_p, seq_len, batch_size)
        Z = zerosfn(T, dim_p, 1, batch_size)
        @views for i in 1:seq_len
            ϕKᵢ, Gᵢ = ϕK[:, i:i, :], Δ[:, i:i, :]
            Z .+= ϕKᵢ
            batched_mul!(Δ_ϕQ[:, i:i, :], Z, Gᵢ)
        end
        fill!(Z, T(0))
        ϕQᵢGᵢ = zerosfn(T, dim_p, 1, batch_size)  # as a buffer
        @views for i in reverse(1:seq_len)
            ϕQᵢ, Gᵢ = ϕQ[:, i:i, :], Δ[:, i:i, :]
            batched_mul!(ϕQᵢGᵢ, ϕQᵢ, Gᵢ)
            Z .+= ϕQᵢGᵢ
            Δ_ϕK[:, i:i, :] .= Z
        end
        (Δ_ϕQ, Δ_ϕK)
    end
    return output, pullback
end