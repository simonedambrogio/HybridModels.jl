using Functors

struct KDCParams{T} <: AbstractKnowledgeDrivenComponent
    params::Vector{T}
    names::Vector{Symbol}
    link::Vector{Function}
end

function Base.show(io::IO, p::KDCParams)
    print(io, "Knowledge-Driven Component\n")
    for (i, (name, value)) in enumerate(zip(p.names, p.params))
        println(io, "  ", name, " = ", p.link[i](value))
    end
end

function KDCParams{T}(params::Vector{T}, names::Vector{Symbol}) where T
    KDCParams{T}(params, names, fill(identity, length(params)))
end

function KDCParams{T}(params::Vector{T}, names::Vector{Symbol}, link::Vector{Function}) where T
    KDCParams{T}(params, names, link)
end

# function KDCParams(nt::NamedTuple)
#     # extract symbols and values from named tuple
#     names = keys(nt) |> collect
#     params = values(nt) |> collect
#     KDCParams{eltype(params)}(params, names, length(params))
# end

@functor KDCParams