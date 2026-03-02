# Null regularizer
export NullRegularizer

@doc raw"""
    NullRegularizer(::Type{T}) where {T <: Real}
    NullRegularizer(lambda::T) where {T <: Real}


Returns the null regularizer, i.e., the function that is identically zero.
```math
h(x) = 0
```

### Arguments
- `T`: The type of zero that is expected to be returned by the regularizer.

In the second constructor, the type of lambda is used to infer the type of zero that is expected to be returned by the regularizer. The value of lambda is ignored.
"""
struct NullRegularizer{T <: Real} <: ShiftedProximableFunction end

NullRegularizer(lambda::T) where {T <: Real} = NullRegularizer{T}()
NullRegularizer(::Type{T}) where {T <: Real} = NullRegularizer{T}()

shifted(h::NullRegularizer{T}, xk::AbstractVector{T}) where {T <: Real} =
  NullRegularizer(T)

function shift!(h::NullRegularizer{T}, xk::AbstractVector{T}) where {T <: Real}
  return h
end

function (h::NullRegularizer{T})(y) where {T <: Real}
  return zero(T)
end

fun_name(ψ::NullRegularizer{T}) where {T <: Real} = "null regularizer"
fun_expr(ψ::NullRegularizer{T}) where {T <: Real} = "t ↦ 0"
fun_params(ψ::NullRegularizer{T}) where {T <: Real} = ""

function Base.show(io::IO, ψ::NullRegularizer{T}) where {T <: Real}
  println(io, "description : ", fun_name(ψ))
  println(io, "expression  : ", fun_expr(ψ))
  println(io, "parameters  : ", fun_params(ψ))
end

function prox!(
  y::AbstractVector{T},
  ψ::NullRegularizer{T},
  q::AbstractVector{T},
  ν::T
) where {T <: Real}
  @assert ν > zero(T)
  y .= q
  return y
end