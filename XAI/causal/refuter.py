import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def causal_refuter(
        model,
        estimand,
        estimate,
        method_name
):
    if method_name == 'bootstrap_refuter':
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name=method_name,
            num_simulations=200,
            random_seed=42,
            sample_size=500,
            show_progress_bar=False,
            n_jobs=-1
        )
        # print(ref)
        return ref

    elif method_name == 'data_subset_refuter':
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name=method_name,
            num_simulations=200,
            subset_fraction=0.8,
            random_seed=42,
            n_jobs=-1
        )
        # print(ref)
        return ref

    elif method_name == 'dummy_outcome_refuter':
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name=method_name,
            random_seed=42,
            num_simulations=200
        )
        # print(ref[0])
        return ref[0]

    elif method_name == 'placebo_treatment_refuter':
        ref = model.refute_estimate(
            estimand,
            estimate,
            num_simulations=200,
            random_seed=42,
            # treatment_names=['pos'],
            n_jobs=-1,
            method_name=method_name
        )
        # print(ref)
        return ref

    else:
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name=method_name,
            num_simulations=200,
            random_state=42,
            n_jobs=-1
        )
        # print(ref)
        return ref