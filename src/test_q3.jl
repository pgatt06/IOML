include(joinpath(@__DIR__, "building_tree.jl"))
include(joinpath(@__DIR__, "utilities.jl"))

using Statistics

struct FeatureCandidate
    left::Int
    right::Int
    operation::Symbol
    label::String
end

function numeric_labels(y)
    classes = unique(y)
    class_to_index = Dict(class => index for (index, class) in enumerate(classes))
    return Float64[class_to_index[label] for label in y]
end

function apply_candidate(candidate::FeatureCandidate, X::Matrix{Float64}; epsilon::Float64 = 1e-6)
    left_column = X[:, candidate.left]
    right_column = X[:, candidate.right]

    if candidate.operation == :sum
        return left_column + right_column
    elseif candidate.operation == :difference
        return abs.(left_column - right_column)
    elseif candidate.operation == :product
        return left_column .* right_column
    elseif candidate.operation == :ratio_lr
        return left_column ./ (right_column .+ epsilon)
    elseif candidate.operation == :ratio_rl
        return right_column ./ (left_column .+ epsilon)
    end

    error("Unknown operation: $(candidate.operation)")
end

function build_feature_candidates(feature_count::Int)
    candidates = FeatureCandidate[]

    for left in 1:(feature_count - 1)
        for right in (left + 1):feature_count
            push!(candidates, FeatureCandidate(left, right, :sum, "x$(left) + x$(right)"))
            push!(candidates, FeatureCandidate(left, right, :difference, "|x$(left) - x$(right)|"))
            push!(candidates, FeatureCandidate(left, right, :product, "x$(left) * x$(right)"))
            push!(candidates, FeatureCandidate(left, right, :ratio_lr, "x$(left) / x$(right)"))
            push!(candidates, FeatureCandidate(left, right, :ratio_rl, "x$(right) / x$(left)"))
        end
    end

    return candidates
end

function safe_correlation(feature::Vector{Float64}, y_numeric::Vector{Float64})
    std(feature) <= 1e-10 && return 0.0
    score = cor(feature, y_numeric)
    return isnan(score) ? 0.0 : abs(score)
end

function normalize_like_train(train_feature::Vector{Float64}, test_feature::Vector{Float64})
    minimum_value = minimum(train_feature)
    maximum_value = maximum(train_feature)
    range_value = maximum_value - minimum_value

    train_scaled = train_feature .- minimum_value
    test_scaled = test_feature .- minimum_value

    if range_value > 0
        train_scaled ./= range_value
        test_scaled ./= range_value
    else
        train_scaled .= 0.0
        test_scaled .= 0.0
    end

    return train_scaled, test_scaled
end

function select_best_candidates(X_train::Matrix{Float64}, Y_train; k_best::Int = 2)
    y_numeric = numeric_labels(Y_train)
    scored_candidates = Vector{Tuple{Float64, FeatureCandidate}}()

    for candidate in build_feature_candidates(size(X_train, 2))
        candidate_feature = apply_candidate(candidate, X_train)
        score = safe_correlation(candidate_feature, y_numeric)
        push!(scored_candidates, (score, candidate))
    end

    sort!(scored_candidates, by = item -> item[1], rev = true)
    selected = scored_candidates[1:min(k_best, length(scored_candidates))]
    return selected
end

function augment_dataset(X_train::Matrix{Float64}, X_test::Matrix{Float64}, selected_candidates)
    augmented_train = copy(X_train)
    augmented_test = copy(X_test)

    for (score, candidate) in selected_candidates
        train_feature = apply_candidate(candidate, X_train)
        test_feature = apply_candidate(candidate, X_test)
        train_scaled, test_scaled = normalize_like_train(train_feature, test_feature)

        augmented_train = hcat(augmented_train, train_scaled)
        augmented_test = hcat(augmented_test, test_scaled)
    end

    return augmented_train, augmented_test
end

function active_split_count(tree)
    tree === nothing && return nothing
    split_count = 2^tree.D - 1
    return count(tree.c[node] == -1 for node in 1:split_count)
end

function summarize_tree_result(tree, X_train, Y_train, X_test, Y_test, classes, resolution_time, gap)
    if tree === nothing
        return (
            train_errors = nothing,
            test_errors = nothing,
            split_nodes = nothing,
            time = round(resolution_time, digits = 1),
            gap = round(gap, digits = 1),
        )
    end

    return (
        train_errors = prediction_errors(tree, X_train, Y_train, classes),
        test_errors = prediction_errors(tree, X_test, Y_test, classes),
        split_nodes = active_split_count(tree),
        time = round(resolution_time, digits = 1),
        gap = round(gap, digits = 1),
    )
end

function print_tree_result(label::String, result)
    if result.train_errors === nothing
        println(label, ": no feasible solution | time = ", result.time, "s | gap = ", result.gap, "%")
        return
    end

    println(
        label,
        ": train/test = ",
        result.train_errors,
        "/",
        result.test_errors,
        " | split nodes = ",
        result.split_nodes,
        " | time = ",
        result.time,
        "s | gap = ",
        result.gap,
        "%",
    )
end

function evaluate_q3_dataset(dataset_name::String; depth::Int = 2, k_best::Int = 2, time_limit::Int = 60)
    dataset = prepare_dataset(dataset_name)

    baseline_tree, _, baseline_time, baseline_gap = build_tree(
        dataset.X_train,
        dataset.Y_train,
        depth,
        dataset.classes;
        multivariate = false,
        time_limit = time_limit,
    )

    selected_candidates = select_best_candidates(dataset.X_train, dataset.Y_train; k_best = k_best)
    augmented_train, augmented_test = augment_dataset(dataset.X_train, dataset.X_test, selected_candidates)

    augmented_tree, _, augmented_time, augmented_gap = build_tree(
        augmented_train,
        dataset.Y_train,
        depth,
        dataset.classes;
        multivariate = false,
        time_limit = time_limit,
    )

    baseline_result = summarize_tree_result(
        baseline_tree,
        dataset.X_train,
        dataset.Y_train,
        dataset.X_test,
        dataset.Y_test,
        dataset.classes,
        baseline_time,
        baseline_gap,
    )
    augmented_result = summarize_tree_result(
        augmented_tree,
        augmented_train,
        dataset.Y_train,
        augmented_test,
        dataset.Y_test,
        dataset.classes,
        augmented_time,
        augmented_gap,
    )

    println("=== Feature augmentation: ", dataset_name, " ===")
    println("Depth: ", depth, " | Added features: ", length(selected_candidates), " | Time limit: ", time_limit, "s")
    print_tree_result("Original dataset", baseline_result)
    print_tree_result("Generated features", augmented_result)
    println("Selected generated features:")
    for (score, candidate) in selected_candidates
        println("  - ", candidate.label, " | score = ", round(score, digits = 3))
    end
    println()
end

function evaluate_precomputed_augmented_dataset(dataset_name::String; depth::Int = 2, time_limit::Int = 60)
    augmented_dataset_name = string(dataset_name, "_augmented")
    original_dataset = prepare_dataset(dataset_name)
    augmented_dataset = prepare_dataset(augmented_dataset_name)

    @assert original_dataset.Y == augmented_dataset.Y "Labels differ between $dataset_name and $augmented_dataset_name"

    baseline_tree, _, baseline_time, baseline_gap = build_tree(
        original_dataset.X_train,
        original_dataset.Y_train,
        depth,
        original_dataset.classes;
        multivariate = false,
        time_limit = time_limit,
    )

    augmented_tree, _, augmented_time, augmented_gap = build_tree(
        augmented_dataset.X_train,
        augmented_dataset.Y_train,
        depth,
        augmented_dataset.classes;
        multivariate = false,
        time_limit = time_limit,
    )

    baseline_result = summarize_tree_result(
        baseline_tree,
        original_dataset.X_train,
        original_dataset.Y_train,
        original_dataset.X_test,
        original_dataset.Y_test,
        original_dataset.classes,
        baseline_time,
        baseline_gap,
    )
    augmented_result = summarize_tree_result(
        augmented_tree,
        augmented_dataset.X_train,
        augmented_dataset.Y_train,
        augmented_dataset.X_test,
        augmented_dataset.Y_test,
        augmented_dataset.classes,
        augmented_time,
        augmented_gap,
    )

    println("=== Precomputed feature augmentation: ", dataset_name, " ===")
    println(
        "Depth: ", depth,
        " | Features: ", size(original_dataset.X_train, 2), " -> ", size(augmented_dataset.X_train, 2),
        " | Time limit: ", time_limit, "s",
    )
    print_tree_result("Original dataset", baseline_result)
    print_tree_result("Precomputed augmented dataset", augmented_result)
    println()
end

function main_q3(; dataset_names = BASE_DATASETS, depth::Int = 2, k_best::Int = 2, time_limit::Int = 60)
    for dataset_name in dataset_names
        evaluate_q3_dataset(dataset_name; depth = depth, k_best = k_best, time_limit = time_limit)
    end
end

function main_q3_precomputed_augmented(;
    dataset_names = PRECOMPUTED_AUGMENTED_DATASETS,
    depth::Int = 2,
    time_limit::Int = 60,
)
    for dataset_name in dataset_names
        evaluate_precomputed_augmented_dataset(dataset_name; depth = depth, time_limit = time_limit)
    end
end

function test_q3(; dataset_names = BASE_DATASETS, depth::Int = 2, k_best::Int = 2, time_limit::Int = 60)
    main_q3(dataset_names = dataset_names, depth = depth, k_best = k_best, time_limit = time_limit)
end
