with open('src/eigenp_utils/single_cell.py', 'r') as f:
    content = f.read()

# Replace the is_categorical check with a more robust one
old_code = """    is_categorical = isinstance(adata.obs[obs_key].dtype, pd.CategoricalDtype) or \\
                     pd.api.types.is_object_dtype(adata.obs[obs_key])"""

new_code = """    is_categorical = isinstance(adata.obs[obs_key].dtype, pd.CategoricalDtype) or \\
                     pd.api.types.is_object_dtype(adata.obs[obs_key]) or \\
                     pd.api.types.is_string_dtype(adata.obs[obs_key])"""

new_content = content.replace(old_code, new_code)

with open('src/eigenp_utils/single_cell.py', 'w') as f:
    f.write(new_content)
