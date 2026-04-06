# Changes
## z-order
z-order in utils.py:
Added a function that receives position array, the size of the box as well as possible option to optimize bit-usage memory.
1. Normalize positions from 0 to 1
2. if bit optimazion is enabled, attempt to find the maximum number of bits needed to interleave all bits
3. Check dimension of pos array
4. Encode either in 1d, 2d or 3d based on dimension and bits needed to interleave
# TODO
## z-order
test it lol
# Questions
## z-order
is

      norm_pos = pos / boxsize

good for memory? Should I just overwrite norm_pos?
How do I receive the precision from the disco_dj object? Do I need to adapt the bit convert to the dtype_num and dtype_c_num?

      z_order = jnp.zeros(x_bits.shape[:-1], dtype=jnp.uint64)

   
