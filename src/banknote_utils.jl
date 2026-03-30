function find_banknote_path()
    file_path = joinpath(@__DIR__, "..", "data", "data_banknote_authentication.txt")
    @assert isfile(file_path) "Banknote dataset not found: $file_path"
    return file_path
end

"""
Load the Banknote Authentication dataset.
Each line contains four numeric features followed by one binary label.
"""
function load_banknote(file_path::AbstractString)
    rows = readlines(file_path)
    values = Float64[]
    labels = Int[]
    feature_count = 4

    for row in rows
        stripped_row = strip(row)
        isempty(stripped_row) && continue

        parts = split(stripped_row, ',')
        length(parts) == feature_count + 1 || continue

        features = try
            parse.(Float64, parts[1:feature_count])
        catch
            continue
        end

        label = try
            parse(Int, parts[end])
        catch
            continue
        end

        append!(values, features)
        push!(labels, label)
    end

    sample_count = length(labels)
    @assert sample_count > 0 "No valid samples were read from $file_path"

    X = reshape(values, feature_count, sample_count)'
    return X, labels
end
