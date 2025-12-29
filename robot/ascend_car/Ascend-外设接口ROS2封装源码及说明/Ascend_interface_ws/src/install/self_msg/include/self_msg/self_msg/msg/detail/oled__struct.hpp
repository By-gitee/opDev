// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from self_msg:msg/Oled.idl
// generated code does not contain a copyright notice

#ifndef SELF_MSG__MSG__DETAIL__OLED__STRUCT_HPP_
#define SELF_MSG__MSG__DETAIL__OLED__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__self_msg__msg__Oled __attribute__((deprecated))
#else
# define DEPRECATED__self_msg__msg__Oled __declspec(deprecated)
#endif

namespace self_msg
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Oled_
{
  using Type = Oled_<ContainerAllocator>;

  explicit Oled_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->type = 0;
      this->x = 0;
      this->y = 0;
      this->show = "";
      this->num = 0ul;
    }
  }

  explicit Oled_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : show(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->type = 0;
      this->x = 0;
      this->y = 0;
      this->show = "";
      this->num = 0ul;
    }
  }

  // field types and members
  using _type_type =
    uint8_t;
  _type_type type;
  using _x_type =
    uint8_t;
  _x_type x;
  using _y_type =
    uint8_t;
  _y_type y;
  using _show_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _show_type show;
  using _num_type =
    uint32_t;
  _num_type num;

  // setters for named parameter idiom
  Type & set__type(
    const uint8_t & _arg)
  {
    this->type = _arg;
    return *this;
  }
  Type & set__x(
    const uint8_t & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const uint8_t & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__show(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->show = _arg;
    return *this;
  }
  Type & set__num(
    const uint32_t & _arg)
  {
    this->num = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    self_msg::msg::Oled_<ContainerAllocator> *;
  using ConstRawPtr =
    const self_msg::msg::Oled_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<self_msg::msg::Oled_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<self_msg::msg::Oled_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      self_msg::msg::Oled_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<self_msg::msg::Oled_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      self_msg::msg::Oled_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<self_msg::msg::Oled_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<self_msg::msg::Oled_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<self_msg::msg::Oled_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__self_msg__msg__Oled
    std::shared_ptr<self_msg::msg::Oled_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__self_msg__msg__Oled
    std::shared_ptr<self_msg::msg::Oled_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Oled_ & other) const
  {
    if (this->type != other.type) {
      return false;
    }
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->show != other.show) {
      return false;
    }
    if (this->num != other.num) {
      return false;
    }
    return true;
  }
  bool operator!=(const Oled_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Oled_

// alias to use template instance with default allocator
using Oled =
  self_msg::msg::Oled_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace self_msg

#endif  // SELF_MSG__MSG__DETAIL__OLED__STRUCT_HPP_
