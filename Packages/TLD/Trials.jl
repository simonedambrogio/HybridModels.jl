using GLMakie
include("Trial.jl")

# ----------- Define Trials ----------- #
struct Trials
    n::Vector{Trial}
end

function Base.show(io::IO, t::Trials)

    print(io, "\n", (t.n[1].s[1].type==1 ? "Left vs Right" : "Attended vs Unattended") )
    print(io, "\nNumber of Trials: ", length(t.n))

end

function GLMakie.plot(trial::Trials)
    f = Figure(size=(1200, 500))

    # Total Number of Visits
    ax1 = CairoMakie.Axis(f[1,1], xlabel="Total Number of Visits", ylabel="Proportion of trials")
    
    visit_n = [(length(trial.visit)>0 ? last(trial.visit) : 0) for trial in trial.n]
    visit = 1:maximum(visit_n)
    count = [sum(visit_n.==visitᵢ) for visitᵢ in visit]
    barplot!(ax1, visit, count/sum(count), color=Makie.wong_colors()[3])
    
    # Average Number of Samples
    ax2 = CairoMakie.Axis(f[1,2], xlabel="Visit Number", ylabel="Average Number of Samples")
    samples = [[sum(trial.visit.==visitᵢ) for trial in trial.n] for visitᵢ in visit]
    mean_samples = mean.(samples)
    barplot!(ax2, visit, mean_samples, color=Makie.wong_colors()[3])

    return f
end
