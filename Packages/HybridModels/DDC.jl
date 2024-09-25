using Functors

struct DDCParams{T, S<:AbstractVector{Symbol}, L<:AbstractVector{<:Function}} <: AbstractDataDrivenComponent
    params::T
    names::S
    link::L
end

function Base.show(io::IO, p::DDCParams)
    print(io, "Data-Driven Component\n")
    if p.params isa AbstractVector
        for (name, value, link) in zip(p.names, p.params, p.link)
            println(io, "  ", name, " = ", link(value))
        end
    else
        println(io, "  ", p.names[1], " = ", p.link[1](p.params))
    end
end

# Constructor for vector parameters
function DDCParams(params::AbstractVector, names::AbstractVector{Symbol}, link::AbstractVector{<:Function})
    DDCParams{typeof(params), typeof(names), typeof(link)}(params, names, link)
end

# Constructor for non-vector parameters
function DDCParams(params::T, names::AbstractVector{Symbol}, link::AbstractVector{<:Function}) where T
    DDCParams{T, typeof(names), typeof(link)}(params, names, link)
end

@functor DDCParams
