using Random

include("struct/tree.jl")
include("banknote_utils.jl")
include("wdbc_utils.jl")

if !isdefined(@__MODULE__, :BASE_DATASETS)
    # The three datasets provided directly in the project statement.
    const BASE_DATASETS = ["iris", "seeds", "wine"]
end
if !isdefined(@__MODULE__, :REPORTED_DATASETS)
    const REPORTED_DATASETS = [
        "iris",
        "seeds",
        "wine",
        "banknote",
        "wdbc",
        "titanic_formatted",
        "diabetes_risk_formatted",
    ]
end
if !isdefined(@__MODULE__, :PRECOMPUTED_AUGMENTED_DATASETS)
    const PRECOMPUTED_AUGMENTED_DATASETS = ["titanic_formatted", "diabetes_risk_formatted"]
end
if !isdefined(@__MODULE__, :Q4_DATASETS)
    const Q4_DATASETS = BASE_DATASETS
end
if !isdefined(@__MODULE__, :DEFAULT_DATASETS)
    const DEFAULT_DATASETS = REPORTED_DATASETS
end
if !isdefined(@__MODULE__, :DEFAULT_DEPTHS)
    const DEFAULT_DEPTHS = 2:4
end
if !isdefined(@__MODULE__, :MERGE_GAMMAS)
    const MERGE_GAMMAS = 0.0:0.2:1.0
end
if !isdefined(@__MODULE__, :ITERATIVE_GAMMAS)
    const ITERATIVE_GAMMAS = 0.0:0.2:0.8
end

"""
Split sample indices into training and test sets.
"""
function train_test_indexes(sample_count::Int, test_ratio::Float64 = 0.2; seed::Int = 1)
    Random.seed!(seed)
    permutation = randperm(sample_count)

    test_count = ceil(Int, sample_count * test_ratio)
    test_indexes = permutation[1:test_count]
    train_indexes = permutation[(test_count + 1):sample_count]

    return train_indexes, test_indexes
end

"""
Count misclassified samples for a fitted tree.
"""
function prediction_errors(tree::Tree, x::Matrix{Float64}, y::AbstractVector, classes::AbstractVector)
    sample_count = size(x, 1)
    feature_count = size(x, 2)
    errors = 0

    for sample_id in 1:sample_count
        node = 1

        for _ in 1:(tree.D + 1)
            if tree.c[node] != -1
                errors += classes[tree.c[node]] != y[sample_id]
                break
            end

            split_value = sum(tree.a[feature_id, node] * x[sample_id, feature_id] for feature_id in 1:feature_count)
            node = split_value - tree.b[node] < 0 ? 2 * node : 2 * node + 1
        end
    end

    return errors
end

"""
Normalize each feature to the [0, 1] interval.
"""
function normalize_features(X::AbstractMatrix{<:Real})
    normalized = Matrix{Float64}(X)

    for feature_id in 1:size(normalized, 2)
        minimum_value = minimum(normalized[:, feature_id])
        maximum_value = maximum(normalized[:, feature_id])
        range_value = maximum_value - minimum_value

        normalized[:, feature_id] .-= minimum_value

        if range_value > 0
            normalized[:, feature_id] ./= range_value
        else
            normalized[:, feature_id] .= 0.0
        end
    end

    return normalized
end

centerData(X) = normalize_features(X)

function centerAndSaveDataSet(X, Y::Vector{Int64}, output_file::String)
    normalized_X = normalize_features(X)

    open(output_file, "w") do output
        println(output, "X = ", normalized_X)
        println(output, "Y = ", Y)
    end
end

"""
Load one of the datasets used in the project and return the data matrix, labels and source path.
"""
function local_dataset_path(dataset_name::String)
    return joinpath(@__DIR__, "..", "data", string(dataset_name, ".txt"))
end

function load_local_txt_dataset(dataset_name::String)
    file_path = local_dataset_path(dataset_name)
    @assert isfile(file_path) "Dataset not found: $file_path"

    module_name = Symbol("ProjectData_", replace(dataset_name, r"[^A-Za-z0-9_]" => "_"))
    dataset_module = Module(module_name)
    Base.include(dataset_module, file_path)

    X = Matrix{Float64}(dataset_module.X)
    Y = collect(dataset_module.Y)
    return X, Y, file_path
end

function load_dataset(dataset_name::String)
    if dataset_name == "banknote"
        file_path = find_banknote_path()
        X, Y = load_banknote(file_path)
        return Matrix{Float64}(X), collect(Y), file_path
    elseif dataset_name == "wdbc"
        file_path = find_wdbc_path()
        X, Y = load_wdbc(file_path)
        return Matrix{Float64}(X), collect(Y), file_path
    elseif isfile(local_dataset_path(dataset_name))
        return load_local_txt_dataset(dataset_name)
    end

    error("Unknown dataset: $dataset_name")
end

"""
Load, normalize and split a dataset with the common protocol used in the report.
"""
function prepare_dataset(dataset_name::String; test_ratio::Float64 = 0.2, seed::Int = 1)
    X, Y, file_path = load_dataset(dataset_name)
    # The report follows this project-wide protocol: normalize features, then apply one fixed train/test split.
    normalized_X = normalize_features(X)
    train_indexes, test_indexes = train_test_indexes(length(Y), test_ratio; seed = seed)

    return (
        name = dataset_name,
        file_path = file_path,
        X = normalized_X,
        Y = Y,
        X_train = normalized_X[train_indexes, :],
        Y_train = Y[train_indexes],
        X_test = normalized_X[test_indexes, :],
        Y_test = Y[test_indexes],
        classes = unique(Y),
    )
end

function print_dataset_header(dataset, time_limit::Int)
    println("=== Dataset ", dataset.name, " ===")
    println(
        "Train: ", size(dataset.X_train, 1),
        " | Test: ", size(dataset.X_test, 1),
        " | Features: ", size(dataset.X_train, 2),
        " | Classes: ", length(dataset.classes),
    )
    # The time limit is interpreted per optimization model, not per dataset.
    println("Time limit per optimization: ", time_limit, "s")
end
