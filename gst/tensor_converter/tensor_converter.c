/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_converter.c
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert media types to tensors (as a filter for other general neural network filters)
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */


/**
 *  @mainpage nnstreamer
 *  @section  intro         Introduction
 *  - Introduction      :   Neural Network Streamer for AI Projects
 *  @section   Program      Program Name
 *  - Program Name      :   nnstreamer
 *  - Program Details   :   It provides a neural network framework connectivities (e.g., tensorflow, caffe) for gstreamer streams.
 *    Efficient Streaming for AI Projects: Neural network models wanted to use efficient and flexible streaming management as well.
 *    Intelligent Media Filters!: Use a neural network model as a media filter / converter.
 *    Composite Models!: Allow to use multiple neural network models in a single stream instance.
 *    Multi Model Intelligence!: Allow to use multiple sources for neural network models.
 *  @section  INOUTPUT      Input/output data
 *  - INPUT             :   None
 *  - OUTPUT            :   None
 *  @section  CREATEINFO    Code information
 *  - Initial date      :   2018/06/14
 *  - Version           :   0.1
 */

/**
 * SECTION:element-tensor_converter
 *
 * A filter that converts media stream to tensor stream for NN frameworks.
 * The output is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! tensor_converter ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <gst/gst.h>
#include <glib.h>

#include "tensor_converter.h"
#include <tensor_meta.h>

GST_DEBUG_CATEGORY_STATIC (gst_tensor_converter_debug);
#define GST_CAT_DEFAULT gst_tensor_converter_debug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT,
  PROP_FORCE_MEMCPY,
};

/* the capabilities of the inputs
 *
 * In v0.0.1, this is "bitmap" image stream
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS
    ("video/x-raw, format = (string) {RGB, BGRx}, views = (int)1, interlace-mode = (string)progressive, framerate = (fraction)[ 0/1, 2147483647/1 ]")
    );

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT));

#define gst_tensor_converter_parent_class parent_class
G_DEFINE_TYPE (GstTensor_Converter, gst_tensor_converter,
    GST_TYPE_BASE_TRANSFORM);

static void gst_tensor_converter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_converter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/* GstBaseTransformer vmethod implementations */
static GstFlowReturn gst_tensor_converter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstFlowReturn gst_tensor_converter_transform_ip (GstBaseTransform *
    trans, GstBuffer * buf);
static GstCaps *gst_tensor_converter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensor_converter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_converter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensor_converter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);
/* GObject vmethod implementations */

/**
 * @brief initialize the tensor_converter's class
 */
static void
gst_tensor_converter_class_init (GstTensor_ConverterClass * g_class)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;
  GstTensor_ConverterClass *klass;

  klass = (GstTensor_ConverterClass *) g_class;
  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_converter_set_property;
  gobject_class->get_property = gst_tensor_converter_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_FORCE_MEMCPY,
      g_param_spec_boolean ("force_memcpy", "Force Memcpy",
          "Disable in-place mode and do memcpy ?", FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple (gstelement_class,
      "Tensor_Converter",
      "Convert media stream to tensor stream",
      "Converts audio or video stream to tensor stream of C-Array for neural network framework filters",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  /* Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensor_converter_transform);
  trans_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_tensor_converter_transform_ip);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_converter_transform_caps);
  trans_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_converter_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_converter_set_caps);

  /* Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_converter_transform_size);
}

/**
 * @brief initialize the new element (G_DEFINE_TYPE requires this)
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_converter_init (GstTensor_Converter * filter)
{
  filter->silent = TRUE;
  filter->tensorConfigured = FALSE;
  filter->negotiated = FALSE;
  filter->removePadding = FALSE;
  filter->disableInPlace = FALSE;
}

/**
 * @brief Set property (gst element vmethod)
 */
static void
gst_tensor_converter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensor_Converter *filter = GST_TENSOR_CONVERTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case PROP_FORCE_MEMCPY:
      filter->disableInPlace = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Get property (gst element vmethod)
 */
static void
gst_tensor_converter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensor_Converter *filter = GST_TENSOR_CONVERTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case PROP_FORCE_MEMCPY:
      g_value_set_boolean (value, filter->disableInPlace);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Determine if we need zero-padding (internal static function)
 * @return 1 if we need to remove stride per row from the stream data. 0 otherwise.
 */
static int
remove_stride_padding_per_row (const gchar * format, int width)
{
  /** @todo The actual list is much longer. fill them (read https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html ) */
  if ((!g_strcmp0 (format, "RGB") || !g_strcmp0 (format, "BGR")
          || !g_strcmp0 (format, "I420")) && (width % 4))
    return 1;
  return 0;
}


/******************************************************************
 * GstElement vmethod implementations
 */

#define return_false_if_fail(x)	\
  ret = (x); \
  if (!ret) \
    return FALSE; \
  ;
/**
 * @brief Configure tensor metadata from sink caps (internal static function)
 * @param caps the sink cap.
 * @param filter "this" pointer to be configured.
 * @return FALSE if error. TRUE if ok.
 */
static gboolean
gst_tensor_converter_configure_tensor (const GstCaps * caps,
    GstTensor_Converter * filter)
{
  GstStructure *structure;
  gint rank;
  tensor_dim dimension;
  gint dim;
  tensor_type type;
  gint framerate_numerator;
  gint framerate_denominator;
  gsize tensorFrameSize;
  gboolean ret;
  const gchar *format;
  int i;

  /* This caps is coming from video/x-raw */
  structure = gst_caps_get_structure (caps, 0);
  rank = 3;                     /* [color-space][height][width] */

  return_false_if_fail (gst_structure_get_int (structure, "width", &dim));
  dimension[1] = dim;
  return_false_if_fail (gst_structure_get_int (structure, "height", &dim));
  dimension[2] = dim;
  return_false_if_fail (gst_structure_get_fraction (structure, "framerate",
          &framerate_numerator, &framerate_denominator));
  type = _NNS_UINT8;            /* Assume color depth per component is 8 bit */

  format = gst_structure_get_string (structure, "format");

  if (!g_strcmp0 (format, "RGB"))
    dimension[0] = 3;           /* R G B */
  else if (!g_strcmp0 (format, "BGRx"))
    dimension[0] = 4;           /* B G R x */
  else {
    err_print ("Format = %s\n", format);
    return FALSE;
  }

  /* Emit Warning if RSTRIDE = RU4 (3BPP) && Width % 4 > 0 */
  /** @todo: Add more conditions! */
  if (remove_stride_padding_per_row (format, dimension[1])) {
    filter->removePadding = TRUE;
  }

  dimension[3] = 1;             /* This is 3-D Tensor */
  tensorFrameSize =
      tensor_element_size[type] * get_tensor_element_count (dimension);
  /* Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html */

  if (filter->tensorConfigured == TRUE) {
    /* It has been already configured. Check if they are consistent */
    if (rank == filter->rank &&
        type == filter->type &&
        framerate_numerator == filter->framerate_numerator &&
        tensorFrameSize == filter->tensorFrameSize &&
        framerate_denominator == filter->framerate_denominator) {
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
        if (dimension[i] != filter->dimension[i]) {
          err_print ("  Dimension %d Mismatch with cached: %d --> %d\n", i,
              dimension[i], filter->dimension[i]);
          return FALSE;
        }
      return TRUE;
    }
    err_print ("  Something's wrong. The tensor metadata is inconsistent.\n");
    return FALSE;
  }

  filter->rank = rank;
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    filter->dimension[i] = dimension[i];
  filter->type = type;
  filter->framerate_numerator = framerate_numerator;
  filter->framerate_denominator = framerate_denominator;
  filter->tensorFrameSize = tensorFrameSize;

  filter->tensorConfigured = TRUE;

  /** @todo Support other types */
  filter->input_media_type = _NNS_VIDEO;
  return TRUE;
}


/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
tensor_converter_init (GstPlugin * tensor_converter)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template tensor_converter' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_converter_debug, "tensor_converter",
      0, "Template tensor_converter");

  return gst_element_register (tensor_converter, "tensor_converter",
      GST_RANK_NONE, GST_TYPE_TENSOR_CONVERTER);
}

/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensor_converter"
#endif

/* gstreamer looks for this structure to register tensor_converters
 *
 * exchange the string 'Template tensor_converter' with your tensor_converter description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_converter,
    "tensor_converter",
    tensor_converter_init,
    VERSION, "LGPL", "GStreamer", "http://gstreamer.net/");

/**
 * @brief copies sink pad buffer to src pad buffer (internal static function)
 * @param filter "this" pointer
 * @param inbuf sink pad buffer
 * @param outbuf src pad buffer
 * @return GST_FLOW_OK if ok. other values represents error
 */
static GstFlowReturn
gst_c2t_transformer_videoframe (GstTensor_Converter *
    filter, GstBuffer * inbuf, GstBuffer * outbuf)
{
  size_t sizeB =
      filter->dimension[0] * filter->dimension[1] * filter->dimension[2] *
      filter->dimension[3];

  g_assert (outbuf);
  g_assert (gst_buffer_get_size (outbuf) >= sizeB);

  if (filter->removePadding == TRUE) {
    int d0, d1;
    unsigned char *srcptr, *destptr;
    unsigned int src_idx = 0, dest_idx = 0;
    size_t size = filter->dimension[0] * filter->dimension[1];
    size_t offset = filter->dimension[0] * filter->dimension[1];
    GstMapInfo src_info, dest_info;

    g_assert (offset % 4);

    /** @todo: We don't know if outbuf is already allocated at this point, yet! */
    g_assert (gst_buffer_get_size (outbuf) >=
        (get_tensor_element_count (filter->dimension) *
            tensor_element_size[filter->type]));

    if (offset % 4)
      offset += 4 - (offset % 4);
    /* Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html */

    gst_buffer_map (inbuf, &src_info, GST_MAP_READ);
    gst_buffer_map (outbuf, &dest_info, GST_MAP_WRITE);

    srcptr = src_info.data;
    destptr = dest_info.data;

    for (d0 = 0; d0 < filter->dimension[3]; d0++) {     /* Supposed to be 0 only */
      g_assert (d0 == 0);
      for (d1 = 0; d1 < filter->dimension[2]; d1++) {   /* Height */
        memcpy (destptr + dest_idx, srcptr + src_idx, size);
        dest_idx += size;
        src_idx += offset;
      }
    }

    gst_buffer_unmap (inbuf, &src_info);
    gst_buffer_unmap (outbuf, &dest_info);

    err_print
        ("\n\n\nYOUR STREAM CONFIGURATION INCURS PERFORMANCE DETERIORATION! (1)\nPlease use 4 x n as image width for inputs.\n\n\n");
    return GST_FLOW_OK;
  } else {
    unsigned char *srcptr, *destptr;
    GstMapInfo src_info, dest_info;

    g_assert (gst_buffer_map (inbuf, &src_info, GST_MAP_READ) == TRUE);
    g_assert (gst_buffer_map (outbuf, &dest_info, GST_MAP_WRITE) == TRUE);

    srcptr = src_info.data;
    destptr = dest_info.data;

    memcpy (destptr, srcptr, sizeB);

    gst_buffer_unmap (inbuf, &src_info);
    gst_buffer_unmap (outbuf, &dest_info);

    return GST_FLOW_OK;
  }
  return GST_FLOW_ERROR;
}

/**
 * @brief non-ip transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensor_converter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstFlowReturn res;
  GstTensor_Converter *filter = GST_TENSOR_CONVERTER_CAST (trans);

  if (G_UNLIKELY (!filter->negotiated))
    goto unknown_format;
  if (G_UNLIKELY (!filter->tensorConfigured))
    goto unknown_tensor;

  switch (filter->input_media_type) {
    case _NNS_VIDEO:
      res = gst_c2t_transformer_videoframe (filter, inbuf, outbuf);
      break;
      /* NOT SUPPORTED */
    case _NNS_AUDIO:
    case _NNS_STRING:
    default:
      err_print ("  Unsupported Media Type (%d)\n", filter->input_media_type);
      goto unknown_type;
  }

  return res;

unknown_format:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_tensor:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("unknown format for tensor"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_type:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("not implemented type of media"));
  return GST_FLOW_NOT_SUPPORTED;
}

/**
 * @brief in-place transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensor_converter_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstTensor_Converter *filter = GST_TENSOR_CONVERTER_CAST (trans);

  if (G_UNLIKELY (!filter->negotiated))
    goto unknown_format;
  if (G_UNLIKELY (!filter->tensorConfigured))
    goto unknown_tensor;

  switch (filter->input_media_type) {
    case _NNS_VIDEO:
      if (filter->removePadding == TRUE) {
        /* Remove zero-padding between rows */
        unsigned char *ptr;
        unsigned int row, d0;
        unsigned int dest_idx = 0, src_idx = 0;
        size_t size = filter->dimension[0] * filter->dimension[1];
        size_t offset = size;
        GstMapInfo info;

        g_assert (offset % 4);
        if (offset % 4)
          offset += 4 - (offset % 4);
        /* Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html */

        gst_buffer_map (buf, &info, GST_MAP_READWRITE);
        ptr = info.data;

        for (d0 = 0; d0 < filter->dimension[3]; d0++) { /* Supposed to be 0 only */
          g_assert (d0 == 0);
          for (row = 0; row < filter->dimension[2]; row++) {    /* Height */
            if (dest_idx != src_idx)
              memmove (ptr + dest_idx, ptr + src_idx, size);
            dest_idx += size;
            src_idx += offset;
          }
        }
        /** @todo: Remove the clutter (reduce the size?) after memcpy. (Check if that's really helpful, first) */
        gst_buffer_unmap (buf, &info);

        err_print
            ("\n\n\nYOUR STREAM CONFIGURATION INCURS PERFORMANCE DETERIORATION! (2)\nPlease use 4 x n as image width for inputs.\n\n\n");
      }
      break;
      /* NOT SUPPORTED */
    case _NNS_AUDIO:
    case _NNS_STRING:
    default:
      err_print ("  Unsupported Media Type (%d)\n", filter->input_media_type);
      goto unknown_type;
  }

  /* DO NOTHING. THIS WORKS AS A PASSTHROUGH. We just remove metadata from video */
  return GST_FLOW_OK;

unknown_format:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_tensor:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("unknown format for tensor"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_type:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("not implemented type of media"));
  return GST_FLOW_NOT_SUPPORTED;
}

/**
 * @brief configure srcpad cap from "proposed" cap. (required vmethod for BaseTransform)
 *
 * @param trans ("this" pointer)
 * @param direction (why do we need this?)
 * @param caps sinkpad cap
 * @param filter this element's cap (don't know specifically.)
 */
static GstCaps *
gst_tensor_converter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstCaps *tmp;
  gboolean ret;
  GstTensor_Converter bogusFilter = { 0 };
  bogusFilter.tensorConfigured = FALSE;
  GstTensor_Converter *obj = GST_TENSOR_CONVERTER_CAST (trans);

  /** @todo: Verify if direction == GST_PAD_SINK means caps is sink pad */
  if (direction == GST_PAD_SINK) {
    GstStructure *structure;
    gchar *str;
    /* Skip verifying if caps is compatible: let's assume sink_factory will do that. */
    /** @todo: Verify if this assumption is correct */

    /** @todo CRITICAL: Handle when caps is in range, not fixed */

    /* Construct bogusFilter from caps (sinkpad) */
    ret = gst_tensor_converter_configure_tensor (caps, &bogusFilter);
    if (ret == FALSE) {
      GstStructure *structure = gst_caps_get_structure (caps, 0);
      gchar *str = gst_structure_to_string (structure);
      gchar str2[2048];
      gchar framerate[1024], width[1024], height[1024], colors[1024];
      const gchar *format;
      int fn = -1, fd, w, h;


      if (TRUE == gst_structure_get_fraction (structure, "framerate", &fn, &fd))
        g_sprintf (framerate, "%d/%d", fn, fd);
      else
        g_sprintf (framerate, "[ 0/1, 2147483647/1 ]");

      if (TRUE == gst_structure_get_int (structure, "width", &w))
        g_sprintf (width, "%d", w);
      else
        g_sprintf (width, "[1, 65535]");

      if (TRUE == gst_structure_get_int (structure, "height", &h))
        g_sprintf (height, "%d", h);
      else
        g_sprintf (height, "[1, 65535]");

      format = gst_structure_get_string (structure, "format");
      if (!g_strcmp0 (format, "RGB"))
        g_sprintf (colors, "3");
      else if (!g_strcmp0 (format, "BGRx"))
        g_sprintf (colors, "4");
      else
        g_sprintf (colors, "{3, 4}");

      debug_print (!obj->silent, "Structure from caps = %s\n", str);
      g_free (str);

      g_sprintf (str2,
          "other/tensor, "
          "rank = (int)3, "
          "type = (string)uint8, "
          "framerate = (fraction) %s, "
          "dim1 = (int) %s, "
          "dim2 = (int) %s, "
          "dim3 = (int) %s, "
          "dim4 = (int) 1", framerate, colors, width, height);
      tmp = gst_caps_from_string (str2);
      debug_print (!obj->silent, "Structure from caps to = %s\n", str2);

      /* If given caps are in range for width/height,
         we cannot configure tensor, however, we may return proper srcpad caps */
      /** @todo: see if the error is from ranging width/height before entering here */
      return tmp;
    }

    g_assert (bogusFilter.tensorConfigured == TRUE);

    /* Construct GstCap (srcpad) from bugusFilter */
    tmp = gst_caps_new_simple ("other/tensor",
        "rank", G_TYPE_INT, bogusFilter.rank,
        "dim1", G_TYPE_INT, bogusFilter.dimension[0],
        "dim2", G_TYPE_INT, bogusFilter.dimension[1],
        "dim3", G_TYPE_INT, bogusFilter.dimension[2],
        "dim4", G_TYPE_INT, bogusFilter.dimension[3],
        "type", G_TYPE_STRING, tensor_element_typename[bogusFilter.type],
        "framerate", GST_TYPE_FRACTION, bogusFilter.framerate_numerator,
        bogusFilter.framerate_denominator, NULL);
    if (filter) {
      GstCaps *tmp2 =
          gst_caps_intersect_full (filter, tmp, GST_CAPS_INTERSECT_FIRST);
      gst_caps_unref (tmp);
      tmp = tmp2;
    }
    if (obj->silent == FALSE) {
      structure = gst_caps_get_structure (caps, 0);
      str = gst_structure_to_string (structure);
      debug_print (TRUE, "From = %s\n", str);
      g_free (str);
      structure = gst_caps_get_structure (tmp, 0);
      str = gst_structure_to_string (structure);
      debug_print (TRUE, "To = %s\n", str);
      g_free (str);
    }

    GST_DEBUG_OBJECT (trans, "SINK transformed %" GST_PTR_FORMAT " into %"
        GST_PTR_FORMAT, caps, tmp);
    return tmp;
  } else if (direction == GST_PAD_SRC) {

    GstStructure *structure;
    gchar *str;

    /* Construct possible GstCap (sinkpad) with src_factory */
    /** @todo This supports video only! */
    GstStaticCaps staticcap =
        GST_STATIC_CAPS
        ("video/x-raw, format = (string){RGB, BGRx}, views = (int)1, "
        "interlace-mode = (string)progressive, "
        "framerate = (fraction)[ 0/1, 2147483647/1 ], "
        "width = (int)[1, 65535], " "height = (int)[1, 65535]");
    tmp = gst_static_caps_get (&staticcap);

    if (obj->silent == FALSE) {
      structure = gst_caps_get_structure (caps, 0);
      str = gst_structure_to_string (structure);
      debug_print (TRUE, "Structure from src = %s\n", str);
      g_free (str);
    }
    if (filter) {
      GstCaps *tmp2;
      if (obj->silent == FALSE) {
        structure = gst_caps_get_structure (filter, 0);
        str = gst_structure_to_string (structure);
        debug_print (TRUE, "Structure from filter = %s\n", str);
        g_free (str);
      }

      tmp2 = gst_caps_intersect_full (filter, tmp, GST_CAPS_INTERSECT_FIRST);

      if (obj->silent == FALSE) {
        structure = gst_caps_get_structure (tmp2, 0);
        str = gst_structure_to_string (structure);
        debug_print (TRUE, "Structure from intersection = %s\n", str);
        g_free (str);
      }

      gst_caps_unref (tmp);
      tmp = tmp2;
    }

    GST_DEBUG_OBJECT (trans, "SRC transformed %" GST_PTR_FORMAT " into %"
        GST_PTR_FORMAT, caps, tmp);
    return tmp;
  }
  /* Neither SRC/SINK? Impossible! */
  err_print ("Direction = %d\n", direction);
  GST_DEBUG_OBJECT (trans, "Error pad direction type. direction: %d",
      direction);
  g_assert (TRUE == FALSE);
  return NULL;
}

/**
 * @brief fixate caps. required vmethod of BaseTransform
 */
static GstCaps *
gst_tensor_converter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstCaps *supposed =
      gst_tensor_converter_transform_caps (trans, direction, caps, NULL);
  GstCaps *result;

  GST_DEBUG_OBJECT (trans, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  result = gst_caps_intersect (othercaps, supposed);
  if (gst_caps_is_empty (result)) {
    gst_caps_unref (result);
    result = othercaps;
  } else {
    gst_caps_unref (othercaps);
  }
  gst_caps_unref (supposed);

  GST_DEBUG_OBJECT (trans, "now fixating %" GST_PTR_FORMAT, result);

  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  if (direction == GST_PAD_SINK) {
    if (gst_caps_is_subset (caps, result)) {
      gst_caps_replace (&result, caps);
    }
  }
  return result;
}

/**
 * @brief set caps. required vmethod of BaseTransform
 */
static gboolean
gst_tensor_converter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  /** This is notifier of cap changes for subclass.
   *  However, we do not have subclass (This is the concrete class)
   */
  GstTensor_Converter *filter = GST_TENSOR_CONVERTER_CAST (trans);
  GstVideoInfo in_info;

  GST_DEBUG_OBJECT (trans, "converting from  %" GST_PTR_FORMAT
      " to %" GST_PTR_FORMAT, incaps, outcaps);

  /** @todo Supports video only */
  /* input caps */
  if (!gst_video_info_from_caps (&in_info, incaps)) {
    err_print ("Cannot set_caps\n");
    return FALSE;
  }

  filter->in_info.video = in_info;
  gst_base_transform_set_in_place (trans,
      (filter->disableInPlace == TRUE) ? FALSE : TRUE);

  filter->negotiated = gst_tensor_converter_configure_tensor (incaps, filter);

  /** @todo Verity if outcaps and filter conf are compatible */
  /** @todo THIS IS REQUIRED TO FILL IN: Return FALSE if filter is not compatible with outcaps */

  return TRUE;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * We cannot directly get the value from size value, we need to review the pad-caps.
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensor_converter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  GstTensor_Converter *filter = GST_TENSOR_CONVERTER_CAST (trans);
  const GstCaps *srccap = (direction == GST_PAD_SINK) ? othercaps : caps;
  tensor_dim dim;
  tensor_type type;
  GstTensor_Filter_CheckStatus ret =
      get_tensor_from_padcap (srccap, dim, &type);

  g_assert ((ret & _TFC_ALL) == _TFC_ALL);

  if (!filter->silent) {
    debug_print (TRUE, "transform_size, direction = %s\n",
        (direction == GST_PAD_SINK) ? "sink" : "src");
    GstStructure *structure = gst_caps_get_structure (caps, 0);
    gchar *str = gst_structure_to_string (structure);
    debug_print (TRUE, "cap = %s\n", str);
    g_free (str);
    structure = gst_caps_get_structure (othercaps, 0);
    str = gst_structure_to_string (structure);
    debug_print (TRUE, "othercap = %s\n", str);
    g_free (str);
  }

  *othersize = get_tensor_element_count (dim) * tensor_element_size[type];

  return TRUE;
}
