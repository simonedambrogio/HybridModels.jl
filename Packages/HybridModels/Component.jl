using Functors
include("AbstractTypes.jl")

struct ComponentParams{T, S<:AbstractVector{Symbol}, L<:AbstractVector{<:Function}} <: AbstractComponent
    params::T
    names::S
    link::L
end


function Base.show(io::IO, ac::AbstractComponent)
    p = ac.params

    if ac isa KDC
        println(io, "\nKnowledge-Driven Component")
    else
        println(io, "\nData-Driven Component")
    end
    
    # Find the maximum length of parameter names
    max_name_length = maximum(length(string(name)) for name in p.names)
    
    if typeof(p.params) <: AbstractVector{<:AbstractFloat}
        for (name, value, link) in zip(p.names, p.params, p.link)
            # Format the name with right-justified padding
            formatted_name = rpad(string(name), max_name_length)
            println(io, "  ", formatted_name, " = ", round(link(value), digits=3))
        end
    else
        println(io, "  ", eltype(p.params))
    end
end

function ComponentParams(params::AbstractVector, names::AbstractVector{Symbol})
    ComponentParams(params, names, fill(identity, length(names)))
end

function ComponentParams(params::AbstractVector, names::AbstractVector{Symbol}, link::AbstractVector{<:Function})
    @assert length(params) == length(names) == length(link) "Lengths of params, names, and link must be equal"
    ComponentParams{typeof(params), typeof(names), typeof(link)}(params, names, link)
end

@functor ComponentParams