using Functors
include("Component.jl");

struct DDC <: AbstractDataDrivenComponent
    params::ComponentParams
end

@functor DDC
