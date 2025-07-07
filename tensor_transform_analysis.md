# Analysis: gsttensor_transform.c Format=Flexible Stream Handling Issue (Updated)

## Problem Description

The `gsttensor_transform.c` element cannot properly handle `format=flexible` streams in flexible→flexible transformations. When flexible streams connect to tensor_transform, it fails during caps negotiation because the output tensor information is not properly populated.

## Root Cause Analysis

### The Real Issue: Incomplete Caps Negotiation for Flexible Inputs (Line 2147)

In `gst_tensor_transform_transform_caps()`:

```c
if (gst_tensors_config_is_flexible (&in_config)) {
  /* output caps is also flexible */
  out_config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
} else {
  for (j = 0; j < in_config.info.num_tensors; j++) {
    in_info = gst_tensors_info_get_nth_info (&in_config.info, j);
    out_info = gst_tensors_info_get_nth_info (&out_config.info, j);

    gst_tensor_transform_convert_dimension (filter, direction,
        j, in_info, out_info);
    if (out_info->type == _NNS_END) {
      /* types cannot be specified */
      is_types_not_fixed = TRUE;
    }
  }
}
```

**The Problem**: When input is flexible, the code:
1. ✅ Sets `out_config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE`
2. ❌ **Skips** tensor dimension/type conversion (`gst_tensor_transform_convert_dimension`)
3. ❌ **Leaves output tensor info uninitialized**

**Result**: During caps negotiation, the output caps have `format=flexible` but no tensor structure information, causing downstream elements to fail or the caps negotiation to not work properly.

### What Should Happen

For flexible inputs, the element should:
1. Set format to flexible ✅
2. **Still populate the tensor structure information** for caps negotiation ❌
3. Handle the actual transformation at runtime with the flexible metadata ✅ (this part works)

## The Fix

The solution is to populate tensor information even for flexible inputs during caps negotiation:

```c
if (gst_tensors_config_is_flexible (&in_config)) {
  /* output caps is also flexible */
  out_config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  
  /* ADDED: Even for flexible, we need to populate tensor info for caps negotiation */
  for (j = 0; j < in_config.info.num_tensors; j++) {
    in_info = gst_tensors_info_get_nth_info (&in_config.info, j);
    out_info = gst_tensors_info_get_nth_info (&out_config.info, j);

    gst_tensor_transform_convert_dimension (filter, direction,
        j, in_info, out_info);
    if (out_info->type == _NNS_END) {
      /* types cannot be specified */
      is_types_not_fixed = TRUE;
    }
  }
} else {
  // ... existing static logic
}
```

## Why This Works

1. **Caps Negotiation**: Provides proper tensor structure information for downstream caps negotiation
2. **Runtime Flexibility**: The flexible format flag ensures that actual tensor info comes from buffer metadata at runtime
3. **Backward Compatibility**: Doesn't change static tensor handling
4. **Flexible→Flexible**: Enables proper flexible-to-flexible transformations

## Alternative Approach: Conservative Fix

If the above approach has any issues, a more conservative fix would be to only populate basic structure info:

```c
if (gst_tensors_config_is_flexible (&in_config)) {
  out_config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  
  /* Set basic info for caps negotiation, but mark types as unfixed */
  out_config.info.num_tensors = in_config.info.num_tensors;
  is_types_not_fixed = TRUE;  // This removes 'types' field from caps
} else {
  // ... existing logic
}
```

## The Actual Issue

The problem is **not** format conversion, but rather **incomplete caps negotiation** for flexible tensors. The element advertises flexible support but doesn't provide enough information during caps negotiation for the pipeline to work properly.

This explains why it might appear to "try to use format=static" - the caps negotiation fails for flexible format due to missing tensor info, causing GStreamer to fall back to other available formats.