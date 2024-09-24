using Functors

struct KDCParams{T} <: AbstractKnowledgeDrivenComponent
    params::Vector{T}
    names::Vector{Symbol}
    n_params::Int
end

function Base.show(io::IO, p::KDCParams)
    print(io, "KDCParams{$(eltype(p.params))}(\n")
    for (name, value) in zip(p.names, p.params)
        println(io, "  ", name, " = ", value)
    end
    print(io, ")")
end

function KDCParams{T}(; kwargs...) where T
    params = T[]
    names = Symbol[]
    for (key, value) in kwargs
        push!(params, value)
        push!(names, key)
    end
    KDCParams{T}(params, names, length(params))
end

function KDCParams{T}(params::Vector{T}, names::Vector{Symbol}) where T
    KDCParams{T}(params, names, length(params))
end


@functor KDCParams