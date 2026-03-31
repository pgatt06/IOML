function find_wdbc_path()
    file_path = joinpath(@__DIR__, "..", "data", "wdbc.data")
    @assert isfile(file_path) "WDBC dataset not found: $file_path"
    return file_path
end

"""
Load the Wisconsin Diagnostic Breast Cancer dataset.
Column 1 is an identifier, column 2 is the diagnosis label, and the remaining 30 columns are features.
"""
function load_wdbc(file_path::AbstractString)
    rows = readlines(file_path)
    values = Float64[]
    labels = Int[]
    feature_count = 30

    for row in rows
        stripped_row = strip(row)
        isempty(stripped_row) && continue

        parts = split(stripped_row, ',')
        length(parts) == feature_count + 2 || continue

        # We map the diagnostic labels to a binary class, keeping the same two-class setting as in the course models.
        label = if strip(parts[2]) == "M"
            1
        elseif strip(parts[2]) == "B"
            0
        else
            continue
        end

        features = try
            parse.(Float64, parts[3:end])
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
