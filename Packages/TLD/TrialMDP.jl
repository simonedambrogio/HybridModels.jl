using Statistics, DataFrames

# ----------- Define Trial MDP ----------- #
struct TrialMDP
    max_dots::Int # Maximum numner of dots (100)
    μ::Vector{Float64} # True underlying number of red dots
    green_dots::Vector{Int64} # Initial uncertanty
    prior::NamedTuple # Brior belief of number of red dots
end;

function TrialMDP()
    # Parameters Bernoulli
    μ = Array{Float64}(undef,2)
    μ[1] = rand(0.1:0.05:0.9)
    Δ = rand([-0.3,-0.2,-0.1,0.1,0.2,0.3])
    μ[2]=μ[1]+Δ
    (μ[2]>0.9 || μ[2]<0.1) && (μ[2] = μ[1]-Δ)
    μ = round.(μ.*100)./100
    # Max Dots
    max_dots = Int64(100)
    # Green Dots
    green_dots = round.(rand(0.05:0.01:0.3,3) * max_dots)
    # Prior
    prior = (; α=1, β=1)

    return TrialMDP(max_dots, μ, green_dots, prior);
end


function TrialMDP(dfr::DataFrameRow)
    
    df = DataFrame(dfr);
    
    TrialMDP(
        100, 
        DataFrames.select(df, r"^ups")     |> df -> vcat([col for col in eachcol(df)]...), 
        DataFrames.select(df, r"^colored") |> df -> vcat([col for col in eachcol(df)]...), 
        (; α=1, β=1)
    )

end
