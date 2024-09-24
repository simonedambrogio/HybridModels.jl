using CSV, DataFrames, ProgressBars
using GLMakie
include("utils.jl");
subj_list = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 30];


behav_data = vcat([
    get_psychopy_data(sbj) |> 
    df -> transform(df, :expTime => (x -> fill(i, length(x))) => :subject)
    for (i, sbj) in ProgressBar(enumerate(subj_list))
]...) |> 
df -> rename(df, "chosenOpt " => "chosenOpt");

# First clean of the data
behav_data_clean = behav_data |> 
## Eliminate trials that time out (marked by trigVal = 7)
df -> groupby(df, [:subject, :trialN]) |> 
df -> vcat([dfg for dfg in df if all(dfg.trigVal .!= 7)]...) |> 
# Make trial number sequential
df -> groupby(df, :subject) |> df -> transform(df, :trialN => makesequential => :trial);

# Get binary trials
behav_data_binary = behav_data_clean |> 
# Get only trials with a blocked option
df -> groupby(df, [:subject, :trial]) |> 
df -> vcat([dfg for dfg in df if first(dfg).ifBlocked && first(dfg).trigVal == 30]...) |> 
# Make trial number sequential
df -> groupby(df, :subject) |> df -> transform(df, :trialN => makesequential => :trial);

# Get trinary trials
behav_data_trinary = behav_data_clean |> 
# Get only trials with no blocked option
df -> groupby(df, [:subject, :trial]) |> 
df -> vcat([dfg for dfg in df if !first(dfg).ifBlocked && first(dfg).trigVal == 30]...) |> 
# Make trial number sequential
df -> groupby(df, :subject) |> df -> transform(df, :trialN => makesequential => :trial);


"""
Binary DataFrame must include the following columns:
    - subject: Integer indicating subject number
    - session: Integer indicating session number
    - trial: Integer indicating trial number
    - time: Float indicating the time of the event (rounded to 4 decimal places)
    - duration: Float indicating the duration of the event (rounded to 2 decimal places)
    - event: String indicating the type of event (e.g., "start button", "reveal green", "start sampling", "switch", "stay", "select", "outcome")
    - mouse_position: String indicating the mouse position ("elsewhere", "left", or "right")
    - choice: Integer indicating the choice (1: left, 0: right)
    - visit: Integer indicating the visit number (starting from 0)
    - n_visits: Integer indicating the total number of visits in the trial
    - n_samples: Integer indicating the total number of samples taken in the trial
    - sample_size: Integer indicating the size of the current sample
    - ups_left: Float indicating the underlying proportion of success for the left option
    - ups_right: Float indicating the underlying proportion of success for the right option
    - red_left: Integer indicating the number of red dots for the left option
    - red_right: Integer indicating the number of red dots for the right option
    - colored_left: Integer indicating the number of colored dots for the left option
    - colored_right: Integer indicating the number of colored dots for the right option
    - colored_blocked: Integer indicating the number of colored dots for the blocked option
""";
# Transform the data to match the required format ---
ct1 = behav_data_binary |>
df -> transform(df,
    # Renaming and recoding columns
    :subject => identity => :subject,
    :subject => identity => :session,
    :trial => identity => :trial,
    :expTime => (x -> round.(x; digits=4)) => :time,
    :sampSize => identity => :sample_size,
    # Converting mouse positions to categorical and numeric values
    :sampOpt => ByRow(mp -> mp=="L" ? "left" : mp=="R" ? "right" : "elsewhere") => :mouse_position,
    :sampOpt => ByRow(mp -> mp=="L" ? 1 : mp=="R" ? 2 : 0) => :mouse_position_num,
    # Recoding choice as 1 (left) or 0 (right)
    :chosenOpt => ByRow(x -> x=="nan" ? missing : x=="L" ? 1 : 0) => :choice
) |> 
# Extract duration events
df -> groupby(df, [:subject, :trial]) |> 
df -> vcat([
    transform(dfg, :time => (x -> round.(vcat(diff(x), 0), digits=2)) => :duration)
    for dfg in df
]...) |> 
# Extract number of red dots
df -> transform(df, 
    [:nRed_L_seen, :mouse_position] => ByRow((r, mp) -> mp=="elsewhere" ? 0 : r) => :red_left,
    [:nRed_R_seen, :mouse_position] => ByRow((r, mp) -> mp=="elsewhere" ? 0 : r) => :red_right
) |> 
# Extract visit, n_visits, n_samples, and fill all trial's rows with values of "choice",
# then extract underling proportion of success (ups)
df -> groupby(df, [:subject, :trial]) |> 
df -> vcat([
    dfg |> 
    df -> transform(df, :choice => (x -> fill(missingomit(dfg.choice), size(dfg,1))) => :choice) |> 
    df -> transform(df, :mouse_position_num => (x -> makesequential(x) .- 1) => :visit) |> 
    df -> transform(df, :visit => (x -> fill(last(x), length(x))) => :n_visits) |> 
    df -> transform(df, :mouse_position => (x -> fill(sum(x .!= "elsewhere"), length(x))) => :n_samples) |>
    df -> transform(df, :true_redProp_L => (x -> fill(nanomit(x)[1], length(x))) => :ups_left) |>
    df -> transform(df, :true_redProp_R => (x -> fill(nanomit(x)[1], length(x))) => :ups_right)
    for dfg in df
]...) |> 
# Filter out trials with n_visits > 0
df -> filter(r -> r.n_visits > 0, df) |>
# Here last visit is not actually a visit, it's just the end of the trial
df -> groupby(df, [:subject, :trial]) |> 
df -> transform(df, 
    [:visit, :n_visits] => ByRow((v, nv) -> v==nv ? NaN : v) => :visit,
    :n_visits => (nv -> nv .- 1) => :n_visits
) |>
df -> transform(df, :visit => (x -> fillnan(x, :backward)) => :visit) |>
# Extract number of green dots
df -> groupby(df, [:subject, :trial]) |> 
df -> vcat([
    transform(dfg, 
    :nGreens_L => (g -> fill(nanomit(g), length(g))) => :nGreens_L,
    :nGreens_R => (g -> fill(nanomit(g), length(g))) => :nGreens_R,
    :nGreens_C => (g -> fill(nanomit(g), length(g))) => :nGreens_C)
    for dfg in df
]...) |> 
df -> groupby(df, [:subject, :trial]) |> 
df -> transform(df, 
    # For colored_left:
    # If visit < 2 and mouse is not on left, use nGreens_L (initial green dots)
    # Otherwise, use revealed_L (actual revealed dots)
    [:revealed_L, :nGreens_L, :visit, :mouse_position] => ByRow((r, g, v, mp) -> v<2 && mp!="left" ? g : r) => :colored_left,
    
    # For colored_right:
    # If visit < 2 and mouse is not on right, use nGreens_R (initial green dots)
    # Otherwise, use revealed_R (actual revealed dots)
    [:revealed_R, :nGreens_R, :visit, :mouse_position] => ByRow((r, g, v, mp) -> v<2 && mp!="right" ? g : r) => :colored_right,
    
    # For colored_blocked:
    # Simply use nGreens_C (center green dots) as is
    :nGreens_C => identity => :colored_blocked
) |> 
# Fill NaN values in colored_left, colored_right, and colored_blocked columns
# with the previous non-NaN value (backward direction)
df -> transform(df, :colored_left => (x -> fillnan(x, :backward)) => :colored_left) |> 
df -> transform(df, :colored_right => (x -> fillnan(x, :backward)) => :colored_right) |> 
df -> transform(df, :colored_blocked => (x -> fillnan(x, :backward)) => :colored_blocked) |> 
# Extract event type
df -> transform(df, :trigVal => ByRow(x -> x==30 ? "start button" : "") => :event) |> 
df -> transform(df, [:trigVal, :event] => ByRow((x, e) -> x==1  ? "reveal green" : e) => :event) |> 
df -> transform(df, [:trigVal, :event] => ByRow((x, e) -> x==2  ? "start sampling" : e) => :event) |> 
df -> transform(df, [:trigVal, :event] => ByRow((x, e) -> x==70  ? "switch" : e) => :event) |> 
df -> transform(df, [:trigVal, :event] => ByRow((x, e) -> x==72  ? "stay" : e) => :event) |> 
df -> transform(df, [:trigVal, :event] => ByRow((x, e) -> x==5 || x==6 ? "select" : e) => :event) |> 
df -> transform(df, [:trigVal, :event] => ByRow((x, e) -> x==20  ? "outcome" : e) => :event) |> 
df -> transform(df, [:trigVal, :event] => ByRow((x, e) -> x==31  ? "iti" : e) => :event) |> 
df -> filter(r -> r.event != "iti", df) |> 
df -> transform(df, [:mouse_position, :event] => ByRow((mp, e) -> e=="select" ? "" : mp) => :mouse_position) |> 
df -> transform(df, :mouse_position => (mp -> fillnan(mp, :backward)) => :mouse_position) |> 
# There are missing events, because there are incorrect trigVals = 71, these missing events are stay or switch events
# identify the missing events, and add them to the event column
df -> transform(df, :mouse_position => (x -> vcat("none", x[1:end-1])) => :mouse_position_previous) |> 
df -> transform(df, [:mouse_position_previous, :mouse_position, :event] => ByRow((mp_previous, mp_current, event) -> event!="" ? event : (mp_current == mp_previous ? "stay" : "switch")) => :event) |> 
# Select relevant columns
df -> select(df, [
    :subject, :session, :trial, :time, :duration, :event, :mouse_position, 
    :choice, :visit, :n_visits, :n_samples, :sample_size,
    :ups_left, :ups_right,
    :red_left, :red_right, :colored_left, :colored_right, :colored_blocked
]);

# Save the data
CSV.write( joinpath(path2data, "preprocessed", "binary", "ct1.csv"), ct1 );
