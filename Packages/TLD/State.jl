using DataFrames, Statistics

# ----------- Define State ----------- #
struct State{N}
    time_step
    red::Vector{N}     # [attended, unattended]
    colored::Vector{N} # [attended, unattended] [red+black]
    gaze::N            # 1: left; 2: right
    visited::Vector{N} # [attended, unattended]
    type::N            # 3: "Attended vs Unattended but Gaze encodes Position (Left vs Right)"
end

function Base.show(io::IO, s::State)

    if s.type==1 # Left vs Right
        str1 = string("Left:  (", s.red[1], ", ", (s.colored[1]-s.red[1])*s.visited[1], ")")
        print(io, str1)
        str2 = s.gaze==1 ? string(" * ")  : string("   ") 
        print(io, str2)
        print(io, " " ^ (17-length(str1)), " N = ", s.colored[1])


        idx=2
        str1 = string("\nRight: (", s.red[idx], ", ", (s.colored[idx]-s.red[idx])*s.visited[idx], ")")
        print(io, str1)
        str2 = s.gaze==idx ? string(" * ")  : string("   ") 
        print(io, str2)
        print(io, " " ^ (17-length(str1)), " N = ", s.colored[idx])
    elseif s.type==2
        str1 = string("\nAttended:   (", s.red[1], ", ", (s.colored[1]-s.red[1])*s.visited[1], ")")
        print(io, str1)
        print(io, " " ^ (24-length(str1)), " N = ", s.colored[1])


        idx=2
        str1 = string("\nUnattended: (", s.red[idx], ", ", (s.colored[idx]-s.red[idx])*s.visited[idx], ")")
        print(io, str1)
        print(io, " " ^ (24-length(str1)), " N = ", s.colored[idx])
    
    elseif s.type==3

        if s.time_step==0
            print(io, "Button to Start")
        else
            position = s.gaze==1 ? ["L", "R"] : s.gaze==2 ? ["R", "L"] : ["", ""]

            str1 = string("\n($(position[1])) Attended:   (", s.red[1], ", ", (s.colored[1]-s.red[1])*s.visited[1], ")")
            print(io, str1)
            print(io, " " ^ (24-length(str1)), " N = ", s.colored[1])


            idx=2
            str1 = string("\n($(position[2])) Unattended: (", s.red[idx], ", ", (s.colored[idx]-s.red[idx])*s.visited[idx], ")")
            print(io, str1)
            print(io, " " ^ (24-length(str1)), " N = ", s.colored[idx])
        end
    end

end

function Base.show(s::State)
    print("time_step: ", s.time_step, "\n")
    print("red:       ", s.red, "\n")
    print("colored:   ", s.colored, "\n")
    print("gaze:      ", s.gaze, "\n")
    print("visited:   ", s.visited, "\n")
    print("type:      ", s.type, "\n")
end

extract_field(s::State) = [s.red... (s.colored-s.red)... s.gaze s.visited... s.type];
Base.:(==)(s1::State, s2::State) = all(extract_field(s1) .== extract_field(s2));

function Statistics.mean(s::State) 
    (α, β) = 1, 1;
    heads = s.red; tails = s.colored-heads
    α = α .+ heads; β = β .+ tails

    mean.(Beta.(α, β)) .|> Float32
end

function s2time(states::Vector{State{N}}) where N
    tm = zeros(Int64, length(states));
    for i in 2:length(states)
        tm[i] = (tm[i-1] + 1) * Int(states[i-1].gaze==states[i].gaze)
    end
    return tm
end;

function State(df::DataFrame)
    length(unique(df.subject)) != 1 && error("The df input can only include 1 subject")
    length(unique(df.session)) != 1 && error("The df input can only include 1 session")
    length(unique(df.trial))   != 1 && error("The df input can only include 1 trial")

    [
        State(
            i, Int[dfr.red_left, dfr.red_right], Int[dfr.colored_left, dfr.colored_right, dfr.colored_blocked], 
            dfr.mouse_position == "elsewhere" ? 0 : dfr.mouse_position=="left" ? 1 : 2, # gaze
            Int[sum(df[1:i,:mouse_position] .== "left") > 0, sum(df[1:i,:mouse_position] .== "right") > 0],
            1
        ) for (i, dfr) in enumerate(eachrow(df))
    ]
end

