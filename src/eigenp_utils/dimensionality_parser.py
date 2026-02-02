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
            print('image_dims : ', image_dims)

            # # Find out which dimensions are reduced by the downstream function
            # dummy_input_shape = tuple(image.shape[image_dims.index(dim)] for dim in target_dims if dim in image_dims)
            # print('dummy_input_shape : ', dummy_input_shape)
            # dummy_input = np.ones(dummy_input_shape, dtype=image.dtype)
            # dummy_output_shape = func(dummy_input).shape
            # print('dummy_output_shape : ', dummy_output_shape)
            

            # reduced_dims = [target_dims[idx] for idx, (i, j) in enumerate(zip(dummy_input.shape, dummy_output_shape)) if i != j]
            # print('reduced_dims : ', reduced_dims)

            # Find out which dimensions are reduced by the downstream function
            dummy_input_shape = tuple(image.shape[image_dims.index(dim)] for dim in target_dims if dim in image_dims)
            dummy_input = np.ones(dummy_input_shape, dtype=image.dtype)
            dummy_output_shape = func(dummy_input).shape
            
            # Determine reduced dimensions
            aligned_input_dims = [target_dims[i] for i in range(len(dummy_input_shape))]
            aligned_output_dims = [target_dims[i] for i in range(len(dummy_output_shape))]
            # reduced_dims = [dim for dim in aligned_input_dims if dim not in aligned_output_dims]
            reduced_dims = []
            output_dim_pointer = -1  # start at the last dimension of dummy_output_shape

            for input_dim in reversed(aligned_input_dims):
                input_size = dummy_input_shape[aligned_input_dims.index(input_dim)]
                
                if output_dim_pointer >= -len(dummy_output_shape) and dummy_output_shape[output_dim_pointer] == input_size:
                    # this dimension exists in the output; decrement the output pointer and continue
                    output_dim_pointer -= 1
                else:
                    # this dimension is missing in the output and therefore is a reduced dimension
                    reduced_dims.append(input_dim)

            reduced_dims = list(reversed(reduced_dims))  # reverse it back to the original order

            # FIX: If the number of output dimensions matches input dimensions, assume resizing (no reduction)
            # regardless of size mismatch.
            if len(dummy_output_shape) == len(dummy_input_shape):
                reduced_dims = []

            print('dummy_input_shape : ', dummy_input_shape)
            print('dummy_output_shape : ', dummy_output_shape)


            # reduced_dims = [dim for idx, dim in enumerate(target_dims) if (idx >= len(dummy_output_shape)) or (dummy_input_shape[idx] != dummy_output_shape[idx])]
            print('reduced_dims : ', reduced_dims)


            # Construct the main dimensions to iterate over:
            main_dims = [dim for dim in image_dims if dim not in reduced_dims]

            # If iterate_dims_arg is None, set it to iterate over all available dimensions excluding target ones
            if iterate_dims_arg is None:
                iterate_over_dims = [dim for dim in main_dims if dim not in target_dims]
                iterate_dims_arg = {dim: '0:{}'.format(image.shape[image_dims.index(dim)]) for dim in iterate_over_dims}

            # Map preserved target dimensions to their new sizes (handling resizing)
            preserved_target_dims = [d for d in target_dims if d not in reduced_dims and d in image_dims]
            new_sizes_map = {}
            if len(preserved_target_dims) == len(dummy_output_shape):
                for d, s in zip(preserved_target_dims, dummy_output_shape):
                    new_sizes_map[d] = s

            # Construct the output shape based on the main_dims:
            output_shape = []
            for dim in main_dims:
                if dim in new_sizes_map:
                    output_shape.append(new_sizes_map[dim])
                else:
                    output_shape.append(image.shape[image_dims.index(dim)])

            print('output_shape : ', output_shape)


            # Adjust the output shape based on iterate_dims:
            for dim, slice_str in iterate_dims_arg.items():
                output_shape[main_dims.index(dim)] = len(range(*parse_slice(slice_str)))

            print('sliced_output_shape : ', output_shape)

            result = np.empty(output_shape, dtype=image.dtype)
            print('result.shape : ',result.shape)

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
