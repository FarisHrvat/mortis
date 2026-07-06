import os

from mortis import (
    MissingROIError,
    check_rois,
    draw_ROIs_for_folder,
    filter_background,
    load_from_folder,
    log1p_transform,
    plot_QC,
    save_spatial_data,
    tic_normalize,
)


def main():
    data_folder = "./data"

    if not os.path.exists(data_folder):
        print(f"Please create '{data_folder}' and add spatial data files.")
        return

    print("\n--- STEP 1: LOADING DATA ---")
    adatas = load_from_folder(data_folder)
    if not adatas:
        return

    print("\n--- STEP 2: CHECKING FOR ROIs ---")
    try:
        check_rois(adatas)
        print("✅ All samples have valid ROIs. Proceeding automatically...")
    except MissingROIError as e:
        print(f"\n🛑 {e}")
        ready_adatas = [a for a in adatas if 'is_tissue' in a.obs]
        raw_adatas = [a for a in adatas if 'is_tissue' not in a.obs]

        fixed_adatas = draw_ROIs_for_folder(raw_adatas)
        adatas = ready_adatas + fixed_adatas

    print("\n--- STEP 3: PREPROCESSING CONFIGURATION ---")
    try:
        user_cutoff = float(input("Enter Fold Change Cutoff (e.g., 1.5): "))
    except ValueError:
        print("Invalid input. Defaulting to 1.5")
        user_cutoff = 1.5

    user_mode = input("Enter processing mode ('sample' or 'group') [default: sample]: ").strip().lower()
    if user_mode not in ['sample', 'group']:
        user_mode = 'sample'

    want_save = input("Do you want to save the final output files? (y/n) [default: y]: ").strip().lower()
    save_flag = False if want_save == 'n' else True

    print("\n--- STEP 4: BACKGROUND REMOVAL ---")
    clean_tissues, qc_stats_list = filter_background(adatas, cutoff=user_cutoff, mode=user_mode)

    print("\n--- STEP 5: NORMALIZATION & QC PLOTS ---")
    final_processed_adatas = []

    for i, (adata, stats) in enumerate(zip(clean_tissues, qc_stats_list)):
        adata = tic_normalize(adata)
        adata = log1p_transform(adata)

        final_processed_adatas.append(adata)
        print(f"\n✅ Sample {i+1} fully processed. Final shape: {adata.shape} (Pixels x Clean Metabolites)")

        print(f"Opening QC Plots for Sample {i+1}...")
        plot_QC(stats, clean_adata=adata, sample_name=f"Sample {i+1}")

    save_spatial_data(final_processed_adatas, save=save_flag)

if __name__ == "__main__":
    main()
