# Changes
The general plan is to convert positions into z-order indices, sort them, and then find a list of neighbours based on a slicing window over these neighbours. Finally, the short range forces can be evaluated over each neighbour list (which always has the same length) and we just need to combine it with long range forces for a full static P3M!
## z-order
z-order in utils.py:
Added a function that receives position array, the size of the box as well as possible option to optimize bit-usage memory.
1. Normalize positions from 0 to 1
2. if bit optimazion is enabled, attempt to find the maximum number of bits needed to interleave all bits
3. Check dimension of pos array
4. Encode either in 1d, 2d or 3d based on dimension and bits needed to interleave

## Neighbour function
Added a calculate_neighbour function that receives a sorted array, a cut off value, as well as a slicing window parameter (alongside with the other simulation params).
1. Create a tuple of window_size neighbours for each particle.
2. Slice over the positions that are sorted by z order indices:
```
max_neighbors = window_size * 2
start = jnp.maximum(0, i - window_size)
slice_size = max_neighbors + 1
window = jax.lax.dynamic_slice(sorted_pos, (start, 0), (slice_size, sorted_pos.shape[-1]))
pos_i = sorted_pos[i]
```
3. Calculate the distances for each window
4. Fill all entries of neighbours that are out of cut off range with a dummy value
# TODO

# Questions
## z-order
is

      norm_pos = pos / boxsize

good for memory? Should I just overwrite norm_pos?
How do I receive the precision from the disco_dj object? Do I need to adapt the bit convert to the dtype_num and dtype_c_num?

      z_order = jnp.zeros(x_bits.shape[:-1], dtype=jnp.uint64)

   
