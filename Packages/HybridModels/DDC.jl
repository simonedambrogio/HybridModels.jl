using Functors
struct DDCParams <: AbstractDataDrivenComponent
    params
    names
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
    DDCParams(params, names)
end

@functor DDCParams
