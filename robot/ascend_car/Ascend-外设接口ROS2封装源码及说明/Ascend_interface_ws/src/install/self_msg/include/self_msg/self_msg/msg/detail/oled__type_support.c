// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from self_msg:msg/Oled.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "self_msg/msg/detail/oled__rosidl_typesupport_introspection_c.h"
#include "self_msg/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "self_msg/msg/detail/oled__functions.h"
#include "self_msg/msg/detail/oled__struct.h"


// Include directives for member types
// Member `show`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  self_msg__msg__Oled__init(message_memory);
}

void self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_fini_function(void * message_memory)
{
  self_msg__msg__Oled__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_message_member_array[5] = {
  {
    "type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(self_msg__msg__Oled, type),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(self_msg__msg__Oled, x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(self_msg__msg__Oled, y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "show",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(self_msg__msg__Oled, show),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(self_msg__msg__Oled, num),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_message_members = {
  "self_msg__msg",  // message namespace
  "Oled",  // message name
  5,  // number of fields
  sizeof(self_msg__msg__Oled),
  self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_message_member_array,  // message members
  self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_init_function,  // function to initialize message memory (memory has to be allocated)
  self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_message_type_support_handle = {
  0,
  &self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_self_msg
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, self_msg, msg, Oled)() {
  if (!self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_message_type_support_handle.typesupport_identifier) {
    self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &self_msg__msg__Oled__rosidl_typesupport_introspection_c__Oled_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
