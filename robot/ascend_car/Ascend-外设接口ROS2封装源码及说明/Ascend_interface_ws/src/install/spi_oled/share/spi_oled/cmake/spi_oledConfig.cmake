# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_spi_oled_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED spi_oled_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(spi_oled_FOUND FALSE)
  elseif(NOT spi_oled_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(spi_oled_FOUND FALSE)
  endif()
  return()
endif()
set(_spi_oled_CONFIG_INCLUDED TRUE)

# output package information
if(NOT spi_oled_FIND_QUIETLY)
  message(STATUS "Found spi_oled: 0.0.0 (${spi_oled_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'spi_oled' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${spi_oled_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(spi_oled_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${spi_oled_DIR}/${_extra}")
endforeach()
