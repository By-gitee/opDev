// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from self_msg:msg/Oled.idl
// generated code does not contain a copyright notice

#ifndef SELF_MSG__MSG__DETAIL__OLED__BUILDER_HPP_
#define SELF_MSG__MSG__DETAIL__OLED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "self_msg/msg/detail/oled__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace self_msg
{

namespace msg
{

namespace builder
{

class Init_Oled_num
{
public:
  explicit Init_Oled_num(::self_msg::msg::Oled & msg)
  : msg_(msg)
  {}
  ::self_msg::msg::Oled num(::self_msg::msg::Oled::_num_type arg)
  {
    msg_.num = std::move(arg);
    return std::move(msg_);
  }

private:
  ::self_msg::msg::Oled msg_;
};

class Init_Oled_show
{
public:
  explicit Init_Oled_show(::self_msg::msg::Oled & msg)
  : msg_(msg)
  {}
  Init_Oled_num show(::self_msg::msg::Oled::_show_type arg)
  {
    msg_.show = std::move(arg);
    return Init_Oled_num(msg_);
  }

private:
  ::self_msg::msg::Oled msg_;
};

class Init_Oled_y
{
public:
  explicit Init_Oled_y(::self_msg::msg::Oled & msg)
  : msg_(msg)
  {}
  Init_Oled_show y(::self_msg::msg::Oled::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_Oled_show(msg_);
  }

private:
  ::self_msg::msg::Oled msg_;
};

class Init_Oled_x
{
public:
  explicit Init_Oled_x(::self_msg::msg::Oled & msg)
  : msg_(msg)
  {}
  Init_Oled_y x(::self_msg::msg::Oled::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_Oled_y(msg_);
  }

private:
  ::self_msg::msg::Oled msg_;
};

class Init_Oled_type
{
public:
  Init_Oled_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Oled_x type(::self_msg::msg::Oled::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_Oled_x(msg_);
  }

private:
  ::self_msg::msg::Oled msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::self_msg::msg::Oled>()
{
  return self_msg::msg::builder::Init_Oled_type();
}

}  // namespace self_msg

#endif  // SELF_MSG__MSG__DETAIL__OLED__BUILDER_HPP_
