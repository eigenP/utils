# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
# ]
# ///
import numpy as np
from functools import wraps
from itertools import product

def parse_slice(s):
    ''' Convert string to a slice or integer index. '''
    parts = s.split(":")
    if len(parts) == 1:
        return (int(parts[0]), int(parts[0]) + 1, 1)
    elif len(parts) == 2:
        return (int(parts[0]), int(parts[1]), 1)
    elif len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise ValueError(f"Invalid slice format: {s}")

def dimensionality_parser(target_dims : str, iterate_dims : dict = None):
    '''
    target_dims  := subset of string 'SCTZYX' implying which dimensions to slice
    iterate_dims := if None, iterate through all available dims (remaining of the target dims)
                    else, a dict that allows slicing per dim, e.g.
                    iterate_dims={'S' :'0', 'C': '0', 'T': '0:3:2'}
    '''
    DIMS_ORDER = 'SCTZYX'

    def decorator(func):
        @wraps(func)
        def wrapper(image, *args, **kwargs):

            # Extract iterate_dims and exclude it from further processing
            iterate_dims_arg = kwargs.pop('iterate_dims', iterate_dims)

            # Get the image's dimensions
            image_dims = DIMS_ORDER[-image.ndim:]
            
            # Find out which dimensions are reduced by the downstream function
            # Step 1: Run on original shape to detect rank preservation and get correct output sizes.
            present_target_dims = [dim for dim in target_dims if dim in image_dims]
            aligned_input_dims = present_target_dims

            original_shape_tuple = tuple(image.shape[image_dims.index(dim)] for dim in target_dims if dim in image_dims)
            original_dummy_input = np.ones(original_shape_tuple, dtype=image.dtype)
            original_output_shape = func(original_dummy_input).shape

            reduced_dims = []
            dim_size_map = {}

            # Rank Preservation Check:
            # If the rank (number of dimensions) is preserved, we assume resizing (all dims kept).
            if len(original_shape_tuple) == len(original_output_shape):
                reduced_dims = []
                dim_size_map = dict(zip(aligned_input_dims, original_output_shape))
            else:
                # Rank Reduced. We need to identify WHICH dimensions were reduced.
                # If input has duplicate sizes (e.g. square), standard matching is ambiguous.
                # We use a unique-dimension dummy to disambiguate.

                # Construct unique shape
                original_shape_list = list(original_shape_tuple)
                unique_shape_list = []
                seen_sizes = set()
                for s in original_shape_list:
                    candidate = s
                    while candidate in seen_sizes:
                        candidate += 1
                    seen_sizes.add(candidate)
                    unique_shape_list.append(candidate)

                unique_dummy_shape = tuple(unique_shape_list)
                unique_dummy_input = np.ones(unique_dummy_shape, dtype=image.dtype)
                unique_output_shape = func(unique_dummy_input).shape

                # Use the size-matching heuristic on the UNIQUE shapes
                output_dim_pointer = -1

                for input_dim in reversed(aligned_input_dims):
                    input_size = unique_dummy_shape[aligned_input_dims.index(input_dim)]

                    if output_dim_pointer >= -len(unique_output_shape) and unique_output_shape[output_dim_pointer] == input_size:
                        # Match found in unique output
                        output_dim_pointer -= 1
                    else:
                        reduced_dims.append(input_dim)

                reduced_dims = list(reversed(reduced_dims))

                # Now map preserved dims to ORIGINAL output sizes
                # The remaining dims in aligned_input_dims correspond to original_output_shape
                preserved_dims = [d for d in aligned_input_dims if d not in reduced_dims]
                if len(preserved_dims) == len(original_output_shape):
                    dim_size_map = dict(zip(preserved_dims, original_output_shape))
                else:
                    # Fallback or weird case where ranks don't match expectation?
                    pass

            # Construct the main dimensions to iterate over:
            main_dims = [dim for dim in image_dims if dim not in reduced_dims]

            # If iterate_dims_arg is None, set it to iterate over all available dimensions excluding target ones
            if iterate_dims_arg is None:
                iterate_over_dims = [dim for dim in main_dims if dim not in target_dims]
                iterate_dims_arg = {dim: '0:{}'.format(image.shape[image_dims.index(dim)]) for dim in iterate_over_dims}

            # Construct the output shape based on the main_dims:
            # Check dim_size_map for resized dimensions
            output_shape = []
            for dim in main_dims:
                if dim in dim_size_map:
                    output_shape.append(dim_size_map[dim])
                else:
                    output_shape.append(image.shape[image_dims.index(dim)])

            # Adjust the output shape based on iterate_dims:
            for dim, slice_str in iterate_dims_arg.items():
                if dim in main_dims:
                    output_shape[main_dims.index(dim)] = len(range(*parse_slice(slice_str)))

            result = np.empty(output_shape, dtype=image.dtype)

            # Use iterate_dims_arg keys directly for ordering
            iterate_dims_order = list(iterate_dims_arg.keys())

            slice_ranges = [range(*parse_slice(iterate_dims_arg[dim])) for dim in iterate_dims_order]

            # Initialize counters for the iterate_dims
            counters = {dim: 0 for dim in iterate_dims_arg.keys()}

            for multi_index in product(*slice_ranges):
                slicer_input = []
                for dim in image_dims:
                    if dim not in iterate_dims_arg:
                        slicer_input.append(slice(None))
                    else:
                        slicer_input.append(multi_index[iterate_dims_order.index(dim)])

                processed_slice = func(image[tuple(slicer_input)], *args, **kwargs)

                slicer_output = []
                for dim in main_dims:
                    if dim in iterate_dims_arg:
                        slicer_output.append(counters[dim])
                    else:
                        slicer_output.append(slice(None))

                result[tuple(slicer_output)] = processed_slice

                # Update the counters
                for dim in iterate_dims_arg.keys():
                    counters[dim] += 1
                    if counters[dim] >= output_shape[main_dims.index(dim)]:
                        counters[dim] = 0


            return result
        return wrapper
    return decorator
