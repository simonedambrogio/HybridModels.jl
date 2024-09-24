using Functors
struct DDCParams <: AbstractDataDrivenComponent
    params
    names
    n_params::Int
end

function Base.show(io::IO, p::DDCParams)
    print(io, "DDCParams{$(eltype(p.params))}(\n")
    for (name, value) in zip(p.names, p.params)
        println(io, "  ", name, " = ", value)
    end
    print(io, ")")
end

function DDCParams(; kwargs...)
    params = []
    names = Symbol[]
    for (key, value) in kwargs
        push!(params, value)
        push!(names, key)
    end
    DDCParams(params, names, length(params))
end

function DDCParams(params::Vector, names::Vector{Symbol})
    DDCParams(params, names, length(params))
end

@functor DDCParams
