import anndata as ad


def inspect_h5ad(file_path):
    print(f"--- Inspecting: {file_path} ---")

    # Load the AnnData object
    adata = ad.read_h5ad(file_path)

    # 1. General Info
    print("\n1. GENERAL STRUCTURE:")
    print(adata)

    # 2. X Matrix Info
    print("\n2. DATA MATRIX (X):")
    print(f"Shape: {adata.X.shape}")
    print(f"Type: {type(adata.X)}")

    # 3. Observation Metadata (Spots/Pixels)
    print("\n3. OBS METADATA (Spot Info):")
    print("Columns:", adata.obs.columns.tolist())
    print(adata.obs.head(3))

    # 4. Variable Metadata (Metabolites/Features)
    print("\n4. VAR METADATA (Metabolite Info):")
    print("Columns:", adata.var.columns.tolist())
    print(adata.var.head(3))

    # 5. Multidimensional Observations (Often where Spatial coords live)
    print("\n5. OBSM KEYS (Multidimensional data):")
    print(list(adata.obsm.keys()))

    # 6. Unstructured Metadata
    print("\n6. UNS KEYS (Unstructured data):")
    print(list(adata.uns.keys()))
    print("-" * 50)

# Replace with the path to your first .h5ad file
inspect_h5ad("1b_P487_Healthy_sez1.h5ad")

# Replace with the path to your second .h5ad file
inspect_h5ad("glass_1_left.h5ad")
