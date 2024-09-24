path2data = @__DIR__;
strSbj(sbj) = "sub-$(lpad(sbj, 2, '0'))";
strBlk(sbj, block) = "tld_py_eeg_s$(lpad(sbj, 2, '0'))_block_$block.csv";


function nanomit(x::AbstractVector) 
    y = filter(!isnan, x);
    return length(y) > 1 ? y : y[1]
end;

function missingomit(x::AbstractVector) 
    y = filter(!ismissing, x);
    return length(y) > 1 ? y : y[1]
end;


# Code up a function get_psychopy_data(sbj) that 
# 1. excludes the first (practice) block
# 2. fixes trigger 71s issues 
# 3. returns a DataFrame that consists of the vectical concatentation of all relevant blocks
function get_psychopy_data(sbj)
    # 1. excludes the first (practice) block
    path2behav = joinpath(path2data, "raw", strSbj(sbj), "behav");
    behavioural_data = [
        begin
            path2csv = joinpath(path2behav, strBlk(sbj, block));
            CSV.read(path2csv, DataFrame) |> filter(r -> !isnan(r.expTime));
        end for block in 2:6
    ] |> x -> vcat(x...);

    # 2. fixes trigger 71s issues
    # println("Before fixing, nrow(behavioural_data) = ", nrow(behavioural_data)) # Fix 71-coding
    idx_71 = findall(behavioural_data.trigVal .== 71); # Find the indices where trigger value is equal to 71
    # Check if each occurrence of 71 comes directly after a 70
    for idx in idx_71
        if idx > 1 && behavioural_data[idx-1, :trigVal] == 70
            behavioural_data[idx, :trigVal] = 72
        end
    end
    # println("After fixing, nrow(behavioural_data) = ", nrow(behavioural_data))

    # 3. returns a DataFrame that consists of the vectical concatentation of all relevant blocks
    return behavioural_data
end 



"""
This function transforms a sequence of integers into a new sequence where:
1. NaN values remain as NaN.
2. For non-NaN values, it assigns incrementing integers starting from 1.
3. The integer increments each time there's a change in the input sequence.
4. If the sequence starts with NaN, the first non-NaN value gets assigned 1.

For example:
- Input:  [NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4]
- Output: [NaN, 1, 1, 2, 3, 4, NaN, 5, 5, 6]

This is useful for converting trial numbers or other sequential data into a 
consistent, incrementing sequence while preserving NaN values.
"""
function makesequential(x::Vector)
    x_transformed = Vector{Union{Float64, Int}}(undef, length(x))
    current_value = isnan(first(x)) ? 0 : 1;
    x_transformed[1] = isnan(first(x)) ? NaN : current_value;
    
    for i in 2:size(x,1)
        if isnan(x[i])
            x_transformed[i] = NaN
        else
            x[i] != x[i-1] && (current_value += 1);
            x_transformed[i] = current_value
        end
    end
    return x_transformed
end;
makesequential(x::SubArray) = makesequential(collect(x));
makesequential(x::Vector{String}) = makesequential(collect(x));


"""
This function fills NaN values in a vector with the last non-NaN value.

For example:
- Input:  [NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4]
- Output: [NaN, 1, 1, 2, 1, 5, 5, 3, 3, 4]

This is useful for filling missing data in a sequence.

try to run:
    hcat([NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4], fillnan_with_last([NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4]))
"""
function fillnan_with_last(x::AbstractVector)
    result = similar(x)
    last_valid = NaN
    for i in eachindex(x)
        if !isnan(x[i])
            last_valid = x[i]
        end
        result[i] = last_valid
    end
    return result
end


"""
This function fills NaN values in a vector with the next non-NaN value.

For example:
- Input:  [NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4]
- Output: [1, 1, 1, 2, 1, 5, 3, 3, 3, 4]

This is useful for filling missing data in a sequence.

Try to run:
    hcat([NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4], fillnan_with_next([NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4]))
"""
function fillnan_with_next(x::AbstractVector)
    result = similar(x)
    next_valid = NaN
    for i in reverse(eachindex(x))
        if !isnan(x[i])
            next_valid = x[i]
        end
        result[i] = next_valid
    end
    return result
end

"""
This function fills NaN values in a vector with the last or next non-NaN value.

For example:
- Input:  [NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4]
- Output: [NaN, 1, 1, 2, 1, 5, 5, 3, 3, 4] (when direction is :backward)
- Output: [1, 1, 1, 2, 1, 5, 3, 3, 3, 4] (when direction is :forward)

This is useful for filling missing data in a sequence.

Try to run:
    hcat([11, NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4], fillnan([11, NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4], :backward))
    hcat([11, NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4], fillnan([11, NaN, 1, 1, 2, 1, 5, NaN, 3, 3, 4], :forward))
"""
function fillnan(x::AbstractVector, direction::Symbol) 
    if direction==:backward 
        return fillnan_with_last(x)
    elseif direction==:forward
        return fillnan_with_next(x)
    else
        error("direction must be :backward or :forward")
    end
end;



# Deal with the Strings
"""
This function fills NaN values in a vector of strings with the last non-NaN value.

For example:
- Input:  ["", "1", "1", "2", "1", "5", "", "3", "3", "4"]
- Output: ["", "1", "1", "2", "1", "5", "5", "3", "3", "4"]

This is useful for filling missing data in a sequence.
"""
function fillnan_with_last(x::Vector{String})
    result = similar(x)
    last_valid = ""
    for i in eachindex(x)
        if !isempty(x[i])
            last_valid = x[i]
        end
        result[i] = isempty(last_valid) ? x[i] : last_valid
    end
    return result
end

"""
This function fills NaN values in a vector of strings with the next non-NaN value.

For example:
- Input:  ["", "1", "1", "2", "1", "5", "", "3", "3", "4"]
- Output: ["1", "1", "1", "2", "1", "5", "3", "3", "3", "4"]

This is useful for filling missing data in a sequence.
"""
function fillnan_with_next(x::Vector{String})
    result = similar(x)
    next_valid = ""
    for i in reverse(eachindex(x))
        if !isempty(x[i])
            next_valid = x[i]
        end
        result[i] = isempty(next_valid) ? x[i] : next_valid
    end
    return result
end

"""
This function fills NaN values in a vector of strings with the last or next non-NaN value.

For example:
- Input:  ["", "1", "1", "2", "1", "5", "", "3", "3", "4"]
- Output: ["", "1", "1", "2", "1", "5", "5", "3", "3", "4"] (when direction is :backward)
- Output: ["1", "1", "1", "2", "1", "5", "3", "3", "3", "4"] (when direction is :forward)

This is useful for filling missing data in a sequence.

try to run:
    hcat(["", "1", "1", "2", "1", "5", "", "3", "3", "4"], fillnan(["", "1", "1", "2", "1", "5", "", "3", "3", "4"], :backward))
    hcat(["", "1", "1", "2", "1", "5", "", "3", "3", "4"], fillnan(["", "1", "1", "2", "1", "5", "", "3", "3", "4"], :forward))
"""
function fillnan(x::Vector{String}, direction::Symbol) 
    if direction == :backward 
        return fillnan_with_last(x)
    elseif direction == :forward
        return fillnan_with_next(x)
    else
        error("direction must be :backward or :forward")
    end
end
