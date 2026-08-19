python kuralscw_processing.py --data-path cwr --save-path KuRALS_CW

# To generate a version matching a smaller deployment SoC's fixed RD buffer size
# (block max-pooled down from the native 124x2048), e.g. 8 Doppler bins x 64 Range
# bins, add --doppler-bins/--range-bins and save to a separate directory so it
# doesn't overwrite the full-resolution dataset above:
# python kuralscw_processing.py --data-path cwr --save-path KuRALS_CW_8x64 --doppler-bins 8 --range-bins 64