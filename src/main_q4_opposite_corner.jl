include(joinpath(@__DIR__, "building_tree.jl"))
include(joinpath(@__DIR__, "utilities.jl"))

using MathOptInterface

const Q4MOI = MathOptInterface

"""
Study opposite-corner and linked-set equalities for univariate trees.

The script uses a dedicated unit-flow formulation:
- each sample always follows one root-to-leaf path,
- the objective only counts correctly classified samples,
- opposite-corner equalities are written on the root left-child flow,
- linked-set equalities are written on terminal class assignments.

It compares three variants on the original data:
- baseline unit-flow formulation,
- all equalities added from the start,
- cutting-plane variant.

It also studies the effect of rounding with the cutting-plane variant.
"""

const DEFAULT_Q4_ROUNDINGS = Union{Nothing, Int}[nothing, 2, 1, 0]

function round_features(X::Matrix{Float64}, digits::Union{Nothing, Int})
    digits === nothing && return copy(X)
    return round.(X, digits = digits)
end

function format_rounding(digits::Union{Nothing, Int})
    return digits === nothing ? "none" : string(digits)
end

function value_key(value::Real; digits::Int = 12)
    return string(round(Float64(value), digits = digits))
end

function vector_key(values::AbstractVector{<:Real}; digits::Int = 12)
    return join((value_key(value; digits = digits) for value in values), "|")
end

function build_box_key(lower_bounds::AbstractVector{<:Real}, upper_bounds::AbstractVector{<:Real}; digits::Int = 12)
    parts = String[]
    for feature_id in eachindex(lower_bounds)
        push!(
            parts,
            string(
                value_key(lower_bounds[feature_id]; digits = digits),
                ":",
                value_key(upper_bounds[feature_id]; digits = digits),
            ),
        )
    end
    return join(parts, "|")
end

function varying_features_between(
    first_point::AbstractVector{<:Real},
    second_point::AbstractVector{<:Real};
    tolerance::Float64 = 1e-9,
)
    varying_features = Int[]
    for feature_id in eachindex(first_point)
        abs(first_point[feature_id] - second_point[feature_id]) > tolerance && push!(varying_features, feature_id)
    end
    return varying_features
end

function corner_signature(
    point::AbstractVector{<:Real},
    lower_bounds::AbstractVector{<:Real},
    upper_bounds::AbstractVector{<:Real},
    varying_features::Vector{Int};
    tolerance::Float64 = 1e-9,
)
    signature = 0

    for fixed_feature in setdiff(collect(eachindex(point)), varying_features)
        abs(point[fixed_feature] - lower_bounds[fixed_feature]) <= tolerance || return nothing
    end

    for (bit_id, feature_id) in enumerate(varying_features)
        if abs(point[feature_id] - lower_bounds[feature_id]) <= tolerance
            continue
        elseif abs(point[feature_id] - upper_bounds[feature_id]) <= tolerance
            signature |= 1 << (bit_id - 1)
        else
            return nothing
        end
    end

    return signature
end

function canonical_opposite_equality(first_pair::Tuple{Int, Int}, second_pair::Tuple{Int, Int})
    ordered_first = first_pair[1] < first_pair[2] ? first_pair : (first_pair[2], first_pair[1])
    ordered_second = second_pair[1] < second_pair[2] ? second_pair : (second_pair[2], second_pair[1])

    if ordered_first < ordered_second
        return (ordered_first[1], ordered_first[2], ordered_second[1], ordered_second[2])
    end
    return (ordered_second[1], ordered_second[2], ordered_first[1], ordered_first[2])
end

function identify_opposite_corner_equalities(X::Matrix{Float64}; tolerance::Float64 = 1e-9)
    sample_count = size(X, 1)
    equalities = NTuple{4, Int}[]
    seen_boxes = Set{String}()
    seen_equalities = Set{NTuple{4, Int}}()

    for first_id in 1:(sample_count - 1)
        for second_id in (first_id + 1):sample_count
            # Two samples define an axis-aligned box; we then search the other corners of the same box in the data.
            lower_bounds = min.(X[first_id, :], X[second_id, :])
            upper_bounds = max.(X[first_id, :], X[second_id, :])
            varying_features = varying_features_between(
                @view(X[first_id, :]),
                @view(X[second_id, :]);
                tolerance = tolerance,
            )
            length(varying_features) < 2 && continue

            box_key = build_box_key(lower_bounds, upper_bounds)
            box_key in seen_boxes && continue
            push!(seen_boxes, box_key)

            signatures_to_ids = Dict{Int, Vector{Int}}()
            for sample_id in 1:sample_count
                signature = corner_signature(
                    @view(X[sample_id, :]),
                    lower_bounds,
                    upper_bounds,
                    varying_features;
                    tolerance = tolerance,
                )
                signature === nothing && continue
                push!(get!(signatures_to_ids, signature, Int[]), sample_id)
            end

            signature_mask = (1 << length(varying_features)) - 1
            opposite_pairs = Tuple{Int, Int}[]
            for signature in keys(signatures_to_ids)
                # Opposite corners correspond to complementary bit patterns on the varying features.
                complement = xor(signature, signature_mask)
                signature < complement || continue
                haskey(signatures_to_ids, complement) || continue

                pair_count = min(length(signatures_to_ids[signature]), length(signatures_to_ids[complement]))
                for pair_id in 1:pair_count
                    push!(
                        opposite_pairs,
                        (signatures_to_ids[signature][pair_id], signatures_to_ids[complement][pair_id]),
                    )
                end
            end

            for first_pair_id in 1:(length(opposite_pairs) - 1)
                for second_pair_id in (first_pair_id + 1):length(opposite_pairs)
                    equality = canonical_opposite_equality(
                        opposite_pairs[first_pair_id],
                        opposite_pairs[second_pair_id],
                    )
                    equality in seen_equalities && continue
                    push!(equalities, equality)
                    push!(seen_equalities, equality)
                end
            end
        end
    end

    return equalities
end

function identify_linked_set_equalities(X::Matrix{Float64}; tolerance::Float64 = 1e-9)
    sample_count = size(X, 1)
    feature_count = size(X, 2)
    equalities = Vector{Tuple{Vector{Int}, Vector{Int}, Int}}()
    seen_equalities = Set{String}()

    for feature_id in 1:feature_count
        groups = Dict{String, Vector{Int}}()

        for sample_id in 1:sample_count
            # A linked set varies along one feature only, so the other coordinates are used as the grouping key.
            key_values = [X[sample_id, other_feature] for other_feature in 1:feature_count if other_feature != feature_id]
            key = vector_key(key_values)
            push!(get!(groups, key, Int[]), sample_id)
        end

        for indices in values(groups)
            values_to_ids = Dict{String, Vector{Int}}()
            value_order = Float64[]

            for sample_id in indices
                feature_value = X[sample_id, feature_id]
                key = value_key(feature_value)
                if !haskey(values_to_ids, key)
                    values_to_ids[key] = Int[]
                    push!(value_order, feature_value)
                end
                push!(values_to_ids[key], sample_id)
            end

            sort!(value_order)
            distinct_keys = [value_key(value) for value in value_order]
            distinct_count = length(distinct_keys)

            distinct_count < 3 && continue
            minimum(length(values_to_ids[key]) for key in distinct_keys) < 2 && continue

            repeat_count = minimum(length(values_to_ids[key]) for key in distinct_keys)
            for copy_id in 1:repeat_count
                # The equality compares one cyclic shift of the ordered values on the active feature.
                left_ids = [values_to_ids[distinct_keys[position]][copy_id] for position in 1:distinct_count]
                right_ids = [
                    values_to_ids[distinct_keys[position == distinct_count ? 1 : position + 1]][copy_id]
                    for position in 1:distinct_count
                ]

                equality_key = string(feature_id, "|", join(left_ids, ","), "|", join(right_ids, ","))
                equality_key in seen_equalities && continue
                push!(equalities, (left_ids, right_ids, feature_id))
                push!(seen_equalities, equality_key)
            end
        end
    end

    return equalities
end

function build_equality_candidates(X::Matrix{Float64}; rounding_digits::Union{Nothing, Int} = nothing)
    rounded_X = round_features(X, rounding_digits)
    # Rounding is only used to reveal repeated coordinates and boxes that are hidden by numerical precision.
    opposite_corner_equalities = identify_opposite_corner_equalities(rounded_X)
    linked_set_equalities = identify_linked_set_equalities(rounded_X)

    return (
        X = rounded_X,
        opposite_corner_equalities = opposite_corner_equalities,
        linked_set_equalities = linked_set_equalities,
    )
end

function compute_univariate_margins(X::Matrix{Float64}; tolerance::Float64 = 1e-4)
    feature_count = size(X, 2)
    sample_count = size(X, 1)
    mu_vector = ones(Float64, feature_count)
    mu_min = 1.0
    mu_max = 0.0

    for feature_id in 1:feature_count
        for first_id in 1:sample_count
            for second_id in (first_id + 1):sample_count
                difference = abs(X[first_id, feature_id] - X[second_id, feature_id])
                difference > tolerance || continue
                mu_vector[feature_id] = min(mu_vector[feature_id], difference)
            end
        end
        mu_min = min(mu_min, mu_vector[feature_id])
        mu_max = max(mu_max, mu_vector[feature_id])
    end

    return mu_vector, mu_min, mu_max
end

function class_index_per_sample(y::AbstractVector, classes::AbstractVector)
    # The objective rewards the class assigned to the true label of each sample.
    return [findfirst(classes .== target) for target in y]
end

function build_unit_flow_model(
    X::Matrix{Float64},
    y::AbstractVector,
    D::Int,
    classes;
    relax_integrality::Bool = false,
    time_limit::Int = -1,
    opposite_equalities::Vector{NTuple{4, Int}} = NTuple{4, Int}[],
    linked_equalities::Vector{Tuple{Vector{Int}, Vector{Int}, Int}} = Tuple{Vector{Int}, Vector{Int}, Int}[],
)
    sample_count = length(y)
    feature_count = size(X, 2)
    class_count = length(classes)
    split_count = 2^D - 1
    leaf_count = 2^D
    total_node_count = split_count + leaf_count

    mu_vector, mu_min, mu_max = compute_univariate_margins(X)
    class_indexes = class_index_per_sample(y, classes)

    model = Model(CPLEX.Optimizer)
    set_silent(model)

    if time_limit != -1
        set_time_limit_sec(model, time_limit)
    end

    if relax_integrality
        # The LP relaxation is used in the cutting-plane loop to detect violated equalities cheaply.
        @variable(model, 0 <= a[1:feature_count, 1:split_count] <= 1, base_name = "a")
        @variable(model, 0 <= c[1:class_count, 1:total_node_count] <= 1, base_name = "c")
        @variable(model, 0 <= u_at[1:sample_count, 1:total_node_count] <= 1, base_name = "u_at")
        @variable(model, 0 <= u_stop[1:sample_count, 1:total_node_count] <= 1, base_name = "u_stop")
        @variable(
            model,
            0 <= u_label[1:sample_count, 1:total_node_count, 1:class_count] <= 1,
            base_name = "u_label",
        )
    else
        @variable(model, a[1:feature_count, 1:split_count], Bin, base_name = "a")
        @variable(model, c[1:class_count, 1:total_node_count], Bin, base_name = "c")
        @variable(model, u_at[1:sample_count, 1:total_node_count], Bin, base_name = "u_at")
        @variable(model, u_stop[1:sample_count, 1:total_node_count], Bin, base_name = "u_stop")
        @variable(model, u_label[1:sample_count, 1:total_node_count, 1:class_count], Bin, base_name = "u_label")
    end
    @variable(model, 0 <= b[1:split_count] <= 1, base_name = "b")

    @constraint(
        model,
        [node in 1:split_count],
        sum(a[feature_id, node] for feature_id in 1:feature_count) +
        sum(c[class_id, node] for class_id in 1:class_count) == 1,
    )
    @constraint(
        model,
        [node in (split_count + 1):total_node_count],
        sum(c[class_id, node] for class_id in 1:class_count) == 1,
    )
    @constraint(
        model,
        [node in 1:split_count],
        b[node] <= sum(a[feature_id, node] for feature_id in 1:feature_count),
    )

    # Unlike formulation F, every sample carries one unit of flow from the root.
    # This is the setting in which the opposite-corner and linked-set equalities are valid.
    @constraint(model, [sample_id in 1:sample_count], u_at[sample_id, 1] == 1)
    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:split_count],
        u_at[sample_id, node] ==
        u_at[sample_id, 2 * node] +
        u_at[sample_id, 2 * node + 1] +
        u_stop[sample_id, node],
    )
    @constraint(
        model,
        [sample_id in 1:sample_count, node in (split_count + 1):total_node_count],
        u_at[sample_id, node] == u_stop[sample_id, node],
    )
    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:total_node_count],
        u_stop[sample_id, node] <= sum(c[class_id, node] for class_id in 1:class_count),
    )
    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:total_node_count],
        sum(u_label[sample_id, node, class_id] for class_id in 1:class_count) == u_stop[sample_id, node],
    )
    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:total_node_count, class_id in 1:class_count],
        u_label[sample_id, node, class_id] <= u_stop[sample_id, node],
    )
    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:total_node_count, class_id in 1:class_count],
        u_label[sample_id, node, class_id] <= c[class_id, node],
    )

    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:split_count],
        # As in CM1, mu enforces a strict left branch when a sample is sent to the left child.
        sum(
            a[feature_id, node] * (X[sample_id, feature_id] + mu_vector[feature_id] - mu_min)
            for feature_id in 1:feature_count
        ) + mu_min <= b[node] + (1 + mu_max) * (1 - u_at[sample_id, 2 * node]),
    )
    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:split_count],
        sum(a[feature_id, node] * X[sample_id, feature_id] for feature_id in 1:feature_count) >=
        b[node] - (1 - u_at[sample_id, 2 * node + 1]),
    )
    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:split_count],
        u_at[sample_id, 2 * node] <= sum(a[feature_id, node] for feature_id in 1:feature_count),
    )
    @constraint(
        model,
        [sample_id in 1:sample_count, node in 1:split_count],
        u_at[sample_id, 2 * node + 1] <= sum(a[feature_id, node] for feature_id in 1:feature_count),
    )

    # At depth 2, opposite-corner equalities are written on the left child of the root.
    for (first_id, second_id, third_id, fourth_id) in opposite_equalities
        @constraint(
            model,
            u_at[first_id, 2] + u_at[second_id, 2] ==
            u_at[third_id, 2] + u_at[fourth_id, 2],
        )
    end

    # Linked-set equalities balance the total terminal assignment of both sets for each class.
    for (left_ids, right_ids, _) in linked_equalities
        @constraint(
            model,
            [class_id in 1:class_count],
            sum(u_label[sample_id, node, class_id] for sample_id in left_ids, node in 1:total_node_count) ==
            sum(u_label[sample_id, node, class_id] for sample_id in right_ids, node in 1:total_node_count),
        )
    end

    @objective(
        model,
        Max,
        sum(
            u_label[sample_id, node, class_indexes[sample_id]]
            for sample_id in 1:sample_count, node in 1:total_node_count
        ),
    )

    return model, (a = a, b = b, c = c, u_at = u_at, u_label = u_label)
end

function safe_node_count(model)
    try
        return Int(round(Q4MOI.get(model, Q4MOI.NodeCount())))
    catch
        try
            return Int(round(Q4MOI.get(backend(model), Q4MOI.NodeCount())))
        catch
            return -1
        end
    end
end

function solve_unit_flow_model(
    X::Matrix{Float64},
    y::AbstractVector,
    D::Int,
    classes;
    relax_integrality::Bool = false,
    time_limit::Int = -1,
    opposite_equalities::Vector{NTuple{4, Int}} = NTuple{4, Int}[],
    linked_equalities::Vector{Tuple{Vector{Int}, Vector{Int}, Int}} = Tuple{Vector{Int}, Vector{Int}, Int}[],
)
    model, variables = build_unit_flow_model(
        X,
        y,
        D,
        classes;
        relax_integrality = relax_integrality,
        time_limit = time_limit,
        opposite_equalities = opposite_equalities,
        linked_equalities = linked_equalities,
    )

    start_time = time()
    optimize!(model)
    resolution_time = time() - start_time

    if primal_status(model) != Q4MOI.FEASIBLE_POINT
        return (
            feasible = false,
            model = model,
            variables = variables,
            objective = nothing,
            bound = nothing,
            gap = nothing,
            resolution_time = resolution_time,
            reported_time = resolution_time,
            node_count = safe_node_count(model),
            tree = nothing,
        )
    end

    objective = objective_value(model)
    bound = objective_bound(model)
    gap = if relax_integrality || termination_status(model) == Q4MOI.OPTIMAL
        0.0
    else
        100.0 * abs(objective - bound) / (abs(objective) + 1e-4)
    end

    tree = nothing
    if !relax_integrality
        total_node_count = size(value.(variables.c), 2)
        class_prediction = Vector{Int}(undef, total_node_count)
        c_values = value.(variables.c)

        for node in 1:total_node_count
            class_id = argmax(c_values[:, node])
            class_prediction[node] = c_values[class_id, node] >= 1.0 - 1e-4 ? class_id : -1
        end

        # The reconstructed tree is only used to report training errors from the integer solution.
        tree = Tree(
            D,
            value.(variables.a),
            class_prediction,
            round.(Int, value.(variables.u_at)),
            X,
        )
    end

    return (
        feasible = true,
        model = model,
        variables = variables,
        objective = objective,
        bound = bound,
        gap = gap,
        resolution_time = resolution_time,
        reported_time = resolution_time,
        node_count = relax_integrality ? 0 : safe_node_count(model),
        tree = tree,
    )
end

function opposite_violation(
    equality::NTuple{4, Int},
    u_values::Matrix{Float64},
)
    first_id, second_id, third_id, fourth_id = equality
    return abs(
        u_values[first_id, 2] +
        u_values[second_id, 2] -
        u_values[third_id, 2] -
        u_values[fourth_id, 2],
    )
end

function linked_violation(
    equality::Tuple{Vector{Int}, Vector{Int}, Int},
    u_label_values,
)
    left_ids, right_ids, _ = equality
    class_count = size(u_label_values, 3)
    total_node_count = size(u_label_values, 2)
    return maximum(
        abs(
            sum(u_label_values[sample_id, node, class_id] for sample_id in left_ids, node in 1:total_node_count) -
            sum(u_label_values[sample_id, node, class_id] for sample_id in right_ids, node in 1:total_node_count),
        )
        for class_id in 1:class_count
    )
end

function solve_with_cutting_plane(
    X::Matrix{Float64},
    y::AbstractVector,
    D::Int,
    classes,
    candidates;
    time_limit::Int,
    violation_tolerance::Float64 = 1e-6,
)
    added_opposite_ids = Int[]
    added_linked_ids = Int[]
    lp_iterations = 0
    lp_total_time = 0.0
    last_lp_result = nothing

    while true
        lp_iterations += 1
        # First solve the current LP relaxation, then add only the equalities violated by that fractional solution.
        last_lp_result = solve_unit_flow_model(
            X,
            y,
            D,
            classes;
            relax_integrality = true,
            time_limit = time_limit,
            opposite_equalities = candidates.opposite_corner_equalities[added_opposite_ids],
            linked_equalities = candidates.linked_set_equalities[added_linked_ids],
        )

        if !last_lp_result.feasible
            break
        end
        lp_total_time += last_lp_result.resolution_time

        u_values = value.(last_lp_result.variables.u_at)
        u_label_values = value.(last_lp_result.variables.u_label)
        new_opposite_ids = Int[]
        for candidate_id in eachindex(candidates.opposite_corner_equalities)
            candidate_id in added_opposite_ids && continue
            violation = opposite_violation(
                candidates.opposite_corner_equalities[candidate_id],
                u_values,
            )
            violation > violation_tolerance && push!(new_opposite_ids, candidate_id)
        end

        new_linked_ids = Int[]
        for candidate_id in eachindex(candidates.linked_set_equalities)
            candidate_id in added_linked_ids && continue
            violation = linked_violation(
                candidates.linked_set_equalities[candidate_id],
                u_label_values,
            )
            violation > violation_tolerance && push!(new_linked_ids, candidate_id)
        end

        isempty(new_opposite_ids) && isempty(new_linked_ids) && break
        append!(added_opposite_ids, new_opposite_ids)
        append!(added_linked_ids, new_linked_ids)
        unique!(added_opposite_ids)
        unique!(added_linked_ids)
    end

    # The final MIP is solved once with the subset of equalities selected by the LP separation loop.
    mip_result = solve_unit_flow_model(
        X,
        y,
        D,
        classes;
        relax_integrality = false,
        time_limit = time_limit,
        opposite_equalities = candidates.opposite_corner_equalities[added_opposite_ids],
        linked_equalities = candidates.linked_set_equalities[added_linked_ids],
    )
    total_reported_time = lp_total_time + mip_result.resolution_time

    return (
        lp_result = last_lp_result,
        mip_result = merge(mip_result, (reported_time = total_reported_time, lp_total_time = lp_total_time)),
        lp_iterations = lp_iterations,
        added_opposite_count = length(added_opposite_ids),
        added_linked_count = length(added_linked_ids),
    )
end

function training_error(tree, X::Matrix{Float64}, y::AbstractVector, classes)
    tree === nothing && return nothing
    return prediction_errors(tree, X, y, classes)
end

function summarize_exact_result(result, X::Matrix{Float64}, y::AbstractVector, classes)
    if !result.feasible
        return (
            objective = nothing,
            lp_objective = nothing,
            train_errors = nothing,
            time = round(result.reported_time, digits = 1),
            gap = nothing,
            node_count = result.node_count,
        )
    end

    return (
        objective = round(result.objective, digits = 3),
        lp_objective = round(result.bound, digits = 3),
        train_errors = training_error(result.tree, X, y, classes),
        time = round(result.reported_time, digits = 1),
        gap = round(result.gap, digits = 3),
        node_count = result.node_count,
    )
end

function evaluate_q4_modes(dataset, candidates; depth::Int, time_limit::Int)
    # Baseline: no extra equality. All: every detected equality. Cutting plane: only violated equalities are kept.
    baseline_lp = solve_unit_flow_model(
        candidates.X,
        dataset.Y_train,
        depth,
        dataset.classes;
        relax_integrality = true,
        time_limit = time_limit,
    )
    baseline_mip = solve_unit_flow_model(
        candidates.X,
        dataset.Y_train,
        depth,
        dataset.classes;
        relax_integrality = false,
        time_limit = time_limit,
    )

    all_lp = solve_unit_flow_model(
        candidates.X,
        dataset.Y_train,
        depth,
        dataset.classes;
        relax_integrality = true,
        time_limit = time_limit,
        opposite_equalities = candidates.opposite_corner_equalities,
        linked_equalities = candidates.linked_set_equalities,
    )
    all_mip = solve_unit_flow_model(
        candidates.X,
        dataset.Y_train,
        depth,
        dataset.classes;
        relax_integrality = false,
        time_limit = time_limit,
        opposite_equalities = candidates.opposite_corner_equalities,
        linked_equalities = candidates.linked_set_equalities,
    )

    cutting_plane = solve_with_cutting_plane(
        candidates.X,
        dataset.Y_train,
        depth,
        dataset.classes,
        candidates;
        time_limit = time_limit,
    )

    return (
        baseline = (
            lp = baseline_lp,
            mip = baseline_mip,
            added_opposite_count = 0,
            added_linked_count = 0,
            lp_iterations = 1,
        ),
        all = (
            lp = all_lp,
            mip = all_mip,
            added_opposite_count = length(candidates.opposite_corner_equalities),
            added_linked_count = length(candidates.linked_set_equalities),
            lp_iterations = 1,
        ),
        cutting_plane = (
            lp = cutting_plane.lp_result,
            mip = cutting_plane.mip_result,
            added_opposite_count = cutting_plane.added_opposite_count,
            added_linked_count = cutting_plane.added_linked_count,
            lp_iterations = cutting_plane.lp_iterations,
        ),
    )
end

function print_mode_summary(dataset_name::String, mode_name::String, summary, candidates)
    mip_result = summarize_exact_result(summary.mip, candidates.X, summary.dataset_y, summary.classes)
    lp_objective = summary.lp.feasible ? round(summary.lp.objective, digits = 3) : "n/a"

    println(
        dataset_name,
        " | mode = ",
        mode_name,
        " | candidates opp./linked = ",
        length(candidates.opposite_corner_equalities),
        "/",
        length(candidates.linked_set_equalities),
        " | added opp./linked = ",
        summary.added_opposite_count,
        "/",
        summary.added_linked_count,
        " | LP = ",
        lp_objective,
        " | train errors = ",
        mip_result.train_errors,
        " | time = ",
        mip_result.time,
        "s | B&B nodes = ",
        mip_result.node_count,
        " | cut rounds = ",
        summary.lp_iterations,
    )
end

function evaluate_rounding_effect(dataset; depth::Int, time_limit::Int, rounding_digits = DEFAULT_Q4_ROUNDINGS)
    results = NamedTuple[]
    for digits in rounding_digits
        candidates = build_equality_candidates(dataset.X_train; rounding_digits = digits)
        cutting_plane = solve_with_cutting_plane(
            candidates.X,
            dataset.Y_train,
            depth,
            dataset.classes,
            candidates;
            time_limit = time_limit,
        )
        mip_result = summarize_exact_result(
            cutting_plane.mip_result,
            candidates.X,
            dataset.Y_train,
            dataset.classes,
        )

        push!(
            results,
            (
                rounding = digits,
                opposite_count = length(candidates.opposite_corner_equalities),
                linked_count = length(candidates.linked_set_equalities),
                added_opposite_count = cutting_plane.added_opposite_count,
                added_linked_count = cutting_plane.added_linked_count,
                lp_objective = cutting_plane.lp_result.feasible ? round(cutting_plane.lp_result.objective, digits = 3) : nothing,
                train_errors = mip_result.train_errors,
                time = mip_result.time,
                node_count = mip_result.node_count,
                lp_iterations = cutting_plane.lp_iterations,
            ),
        )
    end
    return results
end

function print_rounding_summary(dataset_name::String, result)
    println(
        dataset_name,
        " | rounding = ",
        format_rounding(result.rounding),
        " | candidates opp./linked = ",
        result.opposite_count,
        "/",
        result.linked_count,
        " | added opp./linked = ",
        result.added_opposite_count,
        "/",
        result.added_linked_count,
        " | LP = ",
        result.lp_objective,
        " | train errors = ",
        result.train_errors,
        " | time = ",
        result.time,
        "s | B&B nodes = ",
        result.node_count,
        " | cut rounds = ",
        result.lp_iterations,
    )
end

function first_useful_rounding(dataset; rounding_digits = DEFAULT_Q4_ROUNDINGS)
    for digits in rounding_digits
        digits === nothing && continue
        candidates = build_equality_candidates(dataset.X_train; rounding_digits = digits)
        useful = length(candidates.opposite_corner_equalities) + length(candidates.linked_set_equalities)
        # We keep the first rounding level that produces at least one exploitable equality.
        useful > 0 && return digits, candidates
    end
    return nothing, nothing
end

function main_q4_opposite_corner(;
    dataset_names::Vector{String} = collect(Q4_DATASETS),
    rounding_digits::Vector{Union{Nothing, Int}} = DEFAULT_Q4_ROUNDINGS,
    depth::Int = 2,
    time_limit::Int = 180,
    comparison_depth::Union{Nothing, Int} = nothing,
    comparison_time_limit::Union{Nothing, Int} = nothing,
    comparison_multivariate::Bool = false,
)
    comparison_multivariate && error("The opposite-corner study only supports univariate trees.")

    actual_depth = comparison_depth === nothing ? depth : comparison_depth
    actual_time_limit = comparison_time_limit === nothing ? time_limit : comparison_time_limit

    println("=== Opposite-corner and linked-set study ===")
    println("Unit-flow univariate formulation with three variants: baseline, all equalities, and cutting plane.")
    println("Depth = ", actual_depth, " | time limit = ", actual_time_limit, "s")
    println()

    println("=== Comparison without rounding ===")
    for dataset_name in dataset_names
        dataset = prepare_dataset(dataset_name)
        # The first pass studies the raw normalized training set, without any coordinate rounding.
        candidates = build_equality_candidates(dataset.X_train)
        mode_results = evaluate_q4_modes(dataset, candidates; depth = actual_depth, time_limit = actual_time_limit)

        println(
            "Dataset: ",
            dataset_name,
            " | train samples = ",
            size(dataset.X_train, 1),
            " | features = ",
            size(dataset.X_train, 2),
        )

        for (mode_name, result) in [
            ("baseline", mode_results.baseline),
            ("all_equalities", mode_results.all),
            ("cutting_plane", mode_results.cutting_plane),
        ]
            summary = merge(result, (dataset_y = dataset.Y_train, classes = dataset.classes))
            print_mode_summary(dataset_name, mode_name, summary, candidates)
        end
        println()
    end

    println("=== Comparison on the first useful rounded data ===")
    for dataset_name in dataset_names
        dataset = prepare_dataset(dataset_name)
        useful_rounding, candidates = first_useful_rounding(dataset; rounding_digits = rounding_digits)

        if useful_rounding === nothing
            println(dataset_name, " | no rounded version generated opposite-corner or linked-set equalities.")
            println()
            continue
        end

        mode_results = evaluate_q4_modes(dataset, candidates; depth = actual_depth, time_limit = actual_time_limit)
        # This second pass checks whether a light rounding makes the equalities informative on the dataset.
        println(
            "Dataset: ",
            dataset_name,
            " | selected rounding = ",
            format_rounding(useful_rounding),
            " | train samples = ",
            size(dataset.X_train, 1),
            " | features = ",
            size(dataset.X_train, 2),
        )

        for (mode_name, result) in [
            ("baseline", mode_results.baseline),
            ("all_equalities", mode_results.all),
            ("cutting_plane", mode_results.cutting_plane),
        ]
            summary = merge(result, (dataset_y = dataset.Y_train, classes = dataset.classes))
            print_mode_summary(dataset_name, mode_name, summary, candidates)
        end
        println()
    end

    println("=== Rounding with cutting plane ===")
    for dataset_name in dataset_names
        dataset = prepare_dataset(dataset_name)
        # The last pass isolates the effect of the rounding level when the cutting-plane strategy is used.
        rounding_results = evaluate_rounding_effect(
            dataset;
            depth = actual_depth,
            time_limit = actual_time_limit,
            rounding_digits = rounding_digits,
        )

        println(
            "Dataset: ",
            dataset_name,
            " | train samples = ",
            size(dataset.X_train, 1),
            " | features = ",
            size(dataset.X_train, 2),
        )

        for result in rounding_results
            print_rounding_summary(dataset_name, result)
        end
        println()
    end
end
