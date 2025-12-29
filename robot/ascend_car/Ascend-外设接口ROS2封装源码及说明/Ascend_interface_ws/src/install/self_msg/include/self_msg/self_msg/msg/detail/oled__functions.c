// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from self_msg:msg/Oled.idl
// generated code does not contain a copyright notice
#include "self_msg/msg/detail/oled__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `show`
#include "rosidl_runtime_c/string_functions.h"

bool
self_msg__msg__Oled__init(self_msg__msg__Oled * msg)
{
  if (!msg) {
    return false;
  }
  // type
  // x
  // y
  // show
  if (!rosidl_runtime_c__String__init(&msg->show)) {
    self_msg__msg__Oled__fini(msg);
    return false;
  }
  // num
  return true;
}

void
self_msg__msg__Oled__fini(self_msg__msg__Oled * msg)
{
  if (!msg) {
    return;
  }
  // type
  // x
  // y
  // show
  rosidl_runtime_c__String__fini(&msg->show);
  // num
}

bool
self_msg__msg__Oled__are_equal(const self_msg__msg__Oled * lhs, const self_msg__msg__Oled * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // type
  if (lhs->type != rhs->type) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // show
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->show), &(rhs->show)))
  {
    return false;
  }
  // num
  if (lhs->num != rhs->num) {
    return false;
  }
  return true;
}

bool
self_msg__msg__Oled__copy(
  const self_msg__msg__Oled * input,
  self_msg__msg__Oled * output)
{
  if (!input || !output) {
    return false;
  }
  // type
  output->type = input->type;
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // show
  if (!rosidl_runtime_c__String__copy(
      &(input->show), &(output->show)))
  {
    return false;
  }
  // num
  output->num = input->num;
  return true;
}

self_msg__msg__Oled *
self_msg__msg__Oled__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  self_msg__msg__Oled * msg = (self_msg__msg__Oled *)allocator.allocate(sizeof(self_msg__msg__Oled), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(self_msg__msg__Oled));
  bool success = self_msg__msg__Oled__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
self_msg__msg__Oled__destroy(self_msg__msg__Oled * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    self_msg__msg__Oled__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
self_msg__msg__Oled__Sequence__init(self_msg__msg__Oled__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  self_msg__msg__Oled * data = NULL;

  if (size) {
    data = (self_msg__msg__Oled *)allocator.zero_allocate(size, sizeof(self_msg__msg__Oled), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = self_msg__msg__Oled__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        self_msg__msg__Oled__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
self_msg__msg__Oled__Sequence__fini(self_msg__msg__Oled__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      self_msg__msg__Oled__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

self_msg__msg__Oled__Sequence *
self_msg__msg__Oled__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  self_msg__msg__Oled__Sequence * array = (self_msg__msg__Oled__Sequence *)allocator.allocate(sizeof(self_msg__msg__Oled__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = self_msg__msg__Oled__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
self_msg__msg__Oled__Sequence__destroy(self_msg__msg__Oled__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    self_msg__msg__Oled__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
self_msg__msg__Oled__Sequence__are_equal(const self_msg__msg__Oled__Sequence * lhs, const self_msg__msg__Oled__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!self_msg__msg__Oled__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
self_msg__msg__Oled__Sequence__copy(
  const self_msg__msg__Oled__Sequence * input,
  self_msg__msg__Oled__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(self_msg__msg__Oled);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    self_msg__msg__Oled * data =
      (self_msg__msg__Oled *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!self_msg__msg__Oled__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          self_msg__msg__Oled__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!self_msg__msg__Oled__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
