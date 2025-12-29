// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from self_msg:msg/Oled.idl
// generated code does not contain a copyright notice

#ifndef SELF_MSG__MSG__DETAIL__OLED__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define SELF_MSG__MSG__DETAIL__OLED__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "self_msg/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "self_msg/msg/detail/oled__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace self_msg
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_self_msg
cdr_serialize(
  const self_msg::msg::Oled & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_self_msg
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  self_msg::msg::Oled & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_self_msg
get_serialized_size(
  const self_msg::msg::Oled & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_self_msg
max_serialized_size_Oled(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace self_msg

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_self_msg
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, self_msg, msg, Oled)();

#ifdef __cplusplus
}
#endif

#endif  // SELF_MSG__MSG__DETAIL__OLED__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
