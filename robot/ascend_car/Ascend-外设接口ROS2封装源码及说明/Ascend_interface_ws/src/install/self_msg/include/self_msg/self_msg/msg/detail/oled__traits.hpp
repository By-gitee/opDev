// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from self_msg:msg/Oled.idl
// generated code does not contain a copyright notice

#ifndef SELF_MSG__MSG__DETAIL__OLED__TRAITS_HPP_
#define SELF_MSG__MSG__DETAIL__OLED__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "self_msg/msg/detail/oled__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace self_msg
{

namespace msg
{

inline void to_flow_style_yaml(
  const Oled & msg,
  std::ostream & out)
{
  out << "{";
  // member: type
  {
    out << "type: ";
    rosidl_generator_traits::value_to_yaml(msg.type, out);
    out << ", ";
  }

  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: show
  {
    out << "show: ";
    rosidl_generator_traits::value_to_yaml(msg.show, out);
    out << ", ";
  }

  // member: num
  {
    out << "num: ";
    rosidl_generator_traits::value_to_yaml(msg.num, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Oled & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "type: ";
    rosidl_generator_traits::value_to_yaml(msg.type, out);
    out << "\n";
  }

  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: show
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "show: ";
    rosidl_generator_traits::value_to_yaml(msg.show, out);
    out << "\n";
  }

  // member: num
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num: ";
    rosidl_generator_traits::value_to_yaml(msg.num, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Oled & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace self_msg

namespace rosidl_generator_traits
{

[[deprecated("use self_msg::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const self_msg::msg::Oled & msg,
  std::ostream & out, size_t indentation = 0)
{
  self_msg::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use self_msg::msg::to_yaml() instead")]]
inline std::string to_yaml(const self_msg::msg::Oled & msg)
{
  return self_msg::msg::to_yaml(msg);
}

template<>
inline const char * data_type<self_msg::msg::Oled>()
{
  return "self_msg::msg::Oled";
}

template<>
inline const char * name<self_msg::msg::Oled>()
{
  return "self_msg/msg/Oled";
}

template<>
struct has_fixed_size<self_msg::msg::Oled>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<self_msg::msg::Oled>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<self_msg::msg::Oled>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SELF_MSG__MSG__DETAIL__OLED__TRAITS_HPP_
