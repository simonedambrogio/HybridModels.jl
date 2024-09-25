__precompile__(false)

module HybridModels
    
    include("HybridModel.jl")
    include("@hybridmodel.jl")
    
    export HybridModel, KDC, DDC, @hybridmodel, @kdc, @ddc, predict!
end

