// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from self_msg:msg/Oled.idl
// generated code does not contain a copyright notice

#ifndef SELF_MSG__MSG__DETAIL__OLED__STRUCT_H_
#define SELF_MSG__MSG__DETAIL__OLED__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'show'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/Oled in the package self_msg.
typedef struct self_msg__msg__Oled
{
  uint8_t type;
  uint8_t x;
  uint8_t y;
  rosidl_runtime_c__String show;
  uint32_t num;
} self_msg__msg__Oled;

// Struct for a sequence of self_msg__msg__Oled.
typedef struct self_msg__msg__Oled__Sequence
{
  self_msg__msg__Oled * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} self_msg__msg__Oled__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SELF_MSG__MSG__DETAIL__OLED__STRUCT_H_
