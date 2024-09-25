using Flux, Statistics, Distributions, Functors, Optimisers, RobustNeuralNetworks,
    StaticArrays, LoopVectorization, Random, ComponentArrays, DataFrames, DataStructures;
using StatsFuns, StatsBase
include("AbstractTypes.jl");
include("KDC.jl");
include("DDC.jl");

struct HybridModel <: AbstractHybridModel
    kdc::AbstractKnowledgeDrivenComponent
    ddc::AbstractDataDrivenComponent
end

# function Agent(kdc::AbstractKnowledgeDrivenComponent, ddc::AbstractDataDrivenComponent)
#     return Agent{typeof(kdc), typeof(ddc)}(kdc, ddc)
# end
# Add this constructor
# Agent(kdc::AbstractKnowledgeDrivenComponent, ddc::AbstractDataDrivenComponent) = Agent{Float32}(kdc, ddc)

@functor HybridModel