
function(get_system_info SYSTEM_INFO)
  if (UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
  elseif (WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else ()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif ()
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")

  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC}
                  -I ${DDK_PATH}/ddk/ai_ddk_lib/include
                  -I ${DDK_PATH}/ddk/ai_ddk_lib/include/secure
                  -I ${DDK_PATH}/tools/tools_ascendc/json-develop/include
                  -I ${DDK_PATH}/ddk/tikcpp/tikcfw
                  -I ${DDK_PATH}/ddk/tikcpp/tikcfw/impl
                  -I ${DDK_PATH}/ddk/tikcpp/tikcfw/interface
                  -L ${DDK_PATH}/tools/tools_omg/master/lib64/
                  -L ${DDK_PATH}/ddk/tbe/lib64
                  -lai_register -lgraph_host -ltiling_api -lplatform
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()
  execute_process(COMMAND ${proj_env} ${prefix_env} ${DDK_PATH}/tools/tools_ascendc/package/opbuild
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(OUTPUT ${OPINFO_OUTPUT}
      COMMAND mkdir -p ${opinfo_file_path}
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/parse_ini_to_json.py
              ${OPINFO_OPS_INFO} ${OPINFO_OUTPUT}
  )
  add_custom_target(${OPINFO_TARGET} ALL
      DEPENDS ${OPINFO_OUTPUT}
  )
  install(FILES ${OPINFO_OUTPUT}
          DESTINATION ${OPINFO_INSTALL_DIR}
  )
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  file(APPEND ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
       "${OP_TYPE},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n")
endfunction()

function(add_ops_impl_target)
  cmake_parse_arguments(OPIMPL "" "TARGET;OPS_INFO;IMPL_DIR;OUT_DIR;INSTALL_DIR" "OPS_BATCH;OPS_ITERATE" ${ARGN})
  add_custom_command(OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp
      COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py
              ${OPIMPL_OPS_INFO}
              \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\"
              ${OPIMPL_IMPL_DIR}
              ${OPIMPL_OUT_DIR}/dynamic
              ${ASCEND_AUTOGEN_PATH}

      COMMAND rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp
      COMMAND touch ${OPIMPL_OUT_DIR}/.impl_timestamp
      DEPENDS ${OPIMPL_OPS_INFO}
              ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py
  )
  add_custom_target(${OPIMPL_TARGET} ALL
      DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp)

  install(DIRECTORY ${OPIMPL_OUT_DIR}/dynamic
    DESTINATION ${OPIMPL_INSTALL_DIR}
  )
endfunction()

function(add_npu_support_target)
  cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
  get_filename_component(npu_sup_file_path "${NPUSUP_OUT_DIR}" DIRECTORY)
  add_custom_command(OUTPUT ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    COMMAND mkdir -p ${NPUSUP_OUT_DIR}
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/util/gen_ops_filter.sh
            ${NPUSUP_OPS_INFO_DIR}
            ${NPUSUP_OUT_DIR}
  )
  add_custom_target(npu_supported_ops ALL
    DEPENDS ${NPUSUP_OUT_DIR}/npu_supported_ops.json
  )
  install(FILES ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    DESTINATION ${NPUSUP_INSTALL_DIR}
  )
endfunction()

function(add_bin_compile_target)
  cmake_parse_arguments(BINCMP "" "TARGET;OPS_INFO;COMPUTE_UNIT;IMPL_DIR;ADP_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
  file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/src)
  file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/bin)
  file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/gen)
  execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_bin_param_build.py
                          ${BINCMP_OPS_INFO} ${BINCMP_OUT_DIR}/gen ${BINCMP_COMPUTE_UNIT}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("ops binary compile scripts gen info: ${EXEC_INFO}")
    message("ops binary compile scripts gen error: ${EXEC_ERROR}")
    message(FATAL_ERROR "ops binary compile scripts gen failed!")
  endif()
  if (NOT TARGET binary)
    add_custom_target(binary)
  endif()
  add_custom_target(${BINCMP_TARGET}
                    COMMAND cp -r ${BINCMP_IMPL_DIR}/*.* ${BINCMP_OUT_DIR}/src
  )
  add_custom_target(${BINCMP_TARGET}_gen_ops_config
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/insert_simplified_keys.py -p ${BINCMP_OUT_DIR}/bin
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_ops_config.py -p ${BINCMP_OUT_DIR}/bin
                            -s ${BINCMP_COMPUTE_UNIT}
  )
  add_dependencies(binary ${BINCMP_TARGET}_gen_ops_config)
  file(GLOB bin_scripts ${BINCMP_OUT_DIR}/gen/*.sh)
  foreach(bin_script ${bin_scripts})
    get_filename_component(bin_file ${bin_script} NAME_WE)
    string(REPLACE "-" ";" bin_sep ${bin_file})
    list(GET bin_sep 0 op_type)
    list(GET bin_sep 1 op_file)
    list(GET bin_sep 2 op_index)
    if (NOT TARGET ${BINCMP_TARGET}_${op_file}_copy)
      file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/bin/${op_file})
      add_custom_target(${BINCMP_TARGET}_${op_file}_copy
                        COMMAND cp ${BINCMP_ADP_DIR}/${op_file}.py ${BINCMP_OUT_DIR}/src/${op_type}.py
      )
      install(DIRECTORY ${BINCMP_OUT_DIR}/bin/${op_file}
        DESTINATION ${BINCMP_INSTALL_DIR}/${BINCMP_COMPUTE_UNIT} OPTIONAL
      )
      install(FILES ${BINCMP_OUT_DIR}/bin/${op_file}.json
        DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT}/ OPTIONAL
      )
    endif()
    add_custom_target(${BINCMP_TARGET}_${op_file}_${op_index}
                      COMMAND export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} && bash ${bin_script} ${BINCMP_OUT_DIR}/src/${op_type}.py ${BINCMP_OUT_DIR}/bin/${op_file} && echo $(MAKE)
                      WORKING_DIRECTORY ${BINCMP_OUT_DIR}
    )
    add_dependencies(${BINCMP_TARGET}_${op_file}_${op_index} ${BINCMP_TARGET} ${BINCMP_TARGET}_${op_file}_copy)
    add_dependencies(${BINCMP_TARGET}_gen_ops_config ${BINCMP_TARGET}_${op_file}_${op_index})
  endforeach()
  install(FILES ${BINCMP_OUT_DIR}/bin/binary_info_config.json
    DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT} OPTIONAL
  )

  install(DIRECTORY ${BINCMP_OUT_DIR}/bin/${op_file}
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../build_out/kernel/${BINCMP_COMPUTE_UNIT} OPTIONAL
  )
  install(FILES ${BINCMP_OUT_DIR}/bin/binary_info_config.json
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../build_out/kernel/config/${BINCMP_COMPUTE_UNIT} OPTIONAL
  )
  install(FILES ${BINCMP_OUT_DIR}/bin/${op_file}.json
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../build_out/kernel/config/${BINCMP_COMPUTE_UNIT} OPTIONAL
  )

endfunction()

function(add_cross_compile_target)
    cmake_parse_arguments(CROSSMP "" "TARGET;OUT_DIR;INSTALL_DIR" "" ${ARGN})
    add_custom_target(${CROSSMP_TARGET} ALL
                        DEPENDS ${CROSSMP_OUT_DIR}
    )
    install(DIRECTORY ${CROSSMP_OUT_DIR}
            DESTINATION ${CROSSMP_INSTALL_DIR}
    )
endfunction()


function(build_host)
  set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  macro(SUBDIRLIST result curdir)
      file(GLOB children RELATIVE ${curdir} ${curdir}/*)
      set(dirlist "")
      foreach(child ${children})
      if(IS_DIRECTORY ${curdir}/${child})
          list(APPEND dirlist ${child})
      endif()
      endforeach()
      set(${result} ${dirlist})
  endmacro()

  set(CUSTOM_OP_PATH ${ROOT_PATH})

  set(LIB_CUSTOM_OP_SRC "")
  foreach(CUSTOM_OP_PROJ ${CUSTOM_OP_PATH})
      file(GLOB CUSTOM_OP_HOST_SRC ${CUSTOM_OP_PATH}/op_host/*.cpp)
      list(APPEND LIB_CUSTOM_OP_SRC ${CUSTOM_OP_HOST_SRC})
  endforeach()

  foreach(CUSTOM_OP_PROJ ${CUSTOM_OP_PATH})
      file(GLOB CUSTOM_OP_HOST_PLUGIN_SRC ${CUSTOM_OP_PATH}/framework/*/*plugin.cpp)
      list(APPEND LIB_CUSTOM_OP_SRC ${CUSTOM_OP_HOST_PLUGIN_SRC})
  endforeach()

  include_directories(${DDK_PATH}/ddk/ai_ddk_lib/include)
  include_directories(${DDK_PATH}/ddk/ai_ddk_lib/include/secure)
  include_directories(${DDK_PATH}/ddk/tikcpp/tikcfw)
  include_directories(${DDK_PATH}/ddk/tikcpp/tikcfw/impl)
  include_directories(${DDK_PATH}/ddk/tikcpp/tikcfw/interface)

  include_directories(${DDK_PATH}/tools/tools_ascendc/json-develop/include)
  set(OMG_LIB64 ${DDK_PATH}/tools/tools_omg/master/lib64)

  find_library(DDK_LIBRARY ai_register ${OMG_LIB64})
  find_library(PLATFORM platform 
    HINTS ${DDK_PATH}/ddk/tbe/lib64/
    HINTS ${OMG_LIB64})
  
  add_library(custom_op SHARED ${LIB_CUSTOM_OP_SRC})
  target_link_libraries(custom_op PUBLIC
    ${DDK_LIBRARY}
    ${OMG_LIB64}/libtiling_api.a
    ${PLATFORM})

  #custom_op
  install(TARGETS custom_op DESTINATION ${OMG_LIB64})
  install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h DESTINATION ${DDK_PATH}/ddk/ascendc/ops/impl/custom/)
endfunction()

function(build_ddk)
  set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  macro(SUBDIRLIST result curdir)
      file(GLOB children RELATIVE ${curdir} ${curdir}/*)
      set(dirlist "")
      foreach(child ${children})
      if(IS_DIRECTORY ${curdir}/${child})
          list(APPEND dirlist ${child})
      endif()
      endforeach()
      set(${result} ${dirlist})
  endmacro()

  set(CUSTOM_OP_PATH ${ROOT_PATH})

  set(LIB_CUSTOM_OP_SRC "")
  foreach(CUSTOM_OP_PROJ ${CUSTOM_OP_PATH})
      file(GLOB CUSTOM_OP_HOST_SRC ${CUSTOM_OP_PATH}/op_host/*.cpp)
      list(APPEND LIB_CUSTOM_OP_SRC ${CUSTOM_OP_HOST_SRC})
  endforeach()

  include_directories(${DDK_PATH}/ddk/ai_ddk_lib/include)
  include_directories(${DDK_PATH}/ddk/ai_ddk_lib/include/secure)
  include_directories(${DDK_PATH}/ddk/tikcpp/tikcfw)
  include_directories(${DDK_PATH}/ddk/tikcpp/tikcfw/impl)
  include_directories(${DDK_PATH}/ddk/tikcpp/tikcfw/interface)

  if (NOT "${OHOS_PLATFORM}" STREQUAL "")
    include_directories(${CUSTOM_OP_PATH}/scripts)
  endif()

  add_library(ai_register SHARED IMPORTED)

  set_target_properties(ai_register PROPERTIES
      IMPORTED_LOCATION ${DDK_PATH}/tools/tools_ascendc/rom_lib/libai_register.so)

  add_library(graph SHARED IMPORTED)
  set_target_properties(graph PROPERTIES
      IMPORTED_LOCATION ${DDK_PATH}/tools/tools_ascendc/rom_lib/libgraph.so)

  add_library(libc++ SHARED IMPORTED)
  set_target_properties(libc++ PROPERTIES
      IMPORTED_LOCATION ${DDK_PATH}/tools/tools_ascendc/rom_lib/libc++.so)

  add_library(custom_op SHARED
      ${LIB_CUSTOM_OP_SRC})

  target_link_libraries(custom_op PUBLIC
      ai_register graph libc++
      )
endfunction()

function(build_kernel)
  file(GLOB_RECURSE SCRI_FILES "${CMAKE_CURRENT_SOURCE_DIR}/build_out/op_kernel/binary/*/gen/*.sh")
  foreach(file ${SCRI_FILES})
    execute_process(
        COMMAND chmod u+x ${file}
    )
  endforeach()

  foreach(file ${SCRI_FILES})
    FOREACH(FILE_PATH ${file})
        STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${FILE_PATH})
    ENDFOREACH(FILE_PATH)
    set(py_name_list "")
    string(REPLACE "-" ";" py_name_list ${FILE_NAME})
    list(LENGTH py_name_list len)
    math(EXPR index "${len} - 2")
    list(GET py_name_list ${index} py_name)
    set(py_file "${CMAKE_CURRENT_SOURCE_DIR}/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/customize/op_impl/*/tbe/customize_impl/dynamic/${py_name}.py")
    set(result_out "${CMAKE_CURRENT_SOURCE_DIR}/build_out/op_kernel/binary/")
    execute_process(
        COMMAND ${file} ${py_file} ${result_out}
    )
  endforeach()
  execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/kernel.sh ${CMAKE_CURRENT_SOURCE_DIR}/build_out/op_kernel/debug
  )
endfunction()

function(build_opinfo)
  set(OP_INFO "")
  set(HOST_OP_INFO "${CMAKE_CURRENT_SOURCE_DIR}/build_out/npu_custom_opinfo.json")
  foreach(CUSTOM_OP_PROJ ${CMAKE_CURRENT_SOURCE_DIR})
    file(GLOB OP_INFO ${CMAKE_CURRENT_SOURCE_DIR}/build_out/autogen/*ops-info.ini)
  endforeach()

  execute_process(
    COMMAND chmod 777 ${DDK_PATH}/tools/tools_ascendc/package/install_custom_opinfo.py
  )

  foreach(OP_INFO_FILE ${OP_INFO})
     execute_process(
      COMMAND python3 ${DDK_PATH}/tools/tools_ascendc/package/install_custom_opinfo.py -i ${OP_INFO_FILE} -t ${HOST_OP_INFO}
    )
  endforeach()

  if (INSTALL_DEVICES)
    file(GLOB_RECURSE SCRI_FILES_TEST "${CMAKE_CURRENT_SOURCE_DIR}/build_out/op_kernel/debug/*/*/*.json")

    set(input_o_file ${CMAKE_CURRENT_SOURCE_DIR}/build_out/autogen/input_o_file.txt)
    file(APPEND ${input_o_file} "{")
    file(APPEND ${input_o_file} "\n")

    set(last_o_file "")
    list(LENGTH SCRI_FILES_TEST len)
    math(EXPR index "${len} - 1")
    list(GET SCRI_FILES_TEST ${index} last_o_file)

    foreach(file ${SCRI_FILES_TEST})
      FOREACH(FILE_PATH ${file})
          STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${FILE_PATH})
      ENDFOREACH(FILE_PATH)
      if ("${last_o_file}" STREQUAL "${file}")
        set(JSON_CONTENT \"${FILE_NAME}\":\"${file}\")
      else()
        set(JSON_CONTENT \"${FILE_NAME}\":\"${file}\",)
      endif()
      file(APPEND ${input_o_file} ${JSON_CONTENT})
      file(APPEND ${input_o_file} "\n")
    endforeach()
    file(APPEND ${input_o_file} "}")

    set(DDK_OP_INFO "${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/deploying/npu_custom_opinfo.json")
    if (COMPILE)
      execute_process(
        COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/compile
      )
      set(DDK_OP_INFO "${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/compile/npu_custom_opinfo.json")
    else()
      execute_process(
        COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/deploying
      )
    endif()

    foreach(OP_INFO_FILE ${OP_INFO})
      execute_process(
        COMMAND python3 ${DDK_PATH}/tools/tools_ascendc/package/install_custom_opinfo.py -i ${OP_INFO_FILE} -t ${DDK_OP_INFO} -o ${input_o_file}
      )
    endforeach()

  endif()
endfunction()

function(install_ddk)
  set(LIB_CUSTOM_OP_SRC "")
  foreach(CUSTOM_OP_PROJ ${CMAKE_CURRENT_SOURCE_DIR})
      file(GLOB CUSTOM_OP_HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp)
      list(APPEND LIB_CUSTOM_OP_SRC ${CUSTOM_OP_HOST_SRC})
  endforeach()

  if (INSTALL_DEVICES)
    build_kernel()
  endif()

  build_opinfo()

  if (INSTALL_DEVICES)
    if (COMPILE)
      install(FILES ${LIB_CUSTOM_OP_SRC} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/compile/)
      install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/build_out/op_kernel/libascendc_custom_kernels.so DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/compile/)
    else()
      install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/build_out/op_kernel/libascendc_custom_kernels.so DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/deploying/)
      install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/libcustom_op.so DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/deploying/)
    endif()

    install(CODE "execute_process(COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/CMakeFiles)")
    install(CODE "execute_process(COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/build.ninja)")
    install(CODE "execute_process(COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/CMakeCache.txt)")
    install(CODE "execute_process(COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/cmake_install.cmake)")
    install(CODE "execute_process(COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/libcustom_op.so)")
    install(CODE "execute_process(COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/.ninja_deps)")
    install(CODE "execute_process(COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/.ninja_log)")
    install(CODE "execute_process(COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/build_out/devices/Makefile)")
  endif()

  foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    set(omg_install_path ${DDK_PATH}/tools/tools_ascendc/custom_op/${compute_unit})
    execute_process(
        COMMAND mkdir -p ${omg_install_path}
        COMMAND rm -rf ${omg_install_path}/*
        COMMAND mkdir -p ${omg_install_path}/src
        COMMAND mkdir -p ${omg_install_path}/lib
        COMMAND mkdir -p ${omg_install_path}/config
        COMMAND mkdir -p ${omg_install_path}/info
        COMMAND mkdir -p ${DDK_PATH}/ddk/ascendc/ops/impl/custom
    )
    set(custom_py "")
    set(custom_cpp "")
    set(custom_h "")
    foreach(CUSTOM_OP_PROJ ${CMAKE_CURRENT_SOURCE_DIR})
      file(GLOB custom_py ${CMAKE_CURRENT_SOURCE_DIR}/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/customize/op_impl/*/tbe/customize_impl/dynamic/*.py)
      install(FILES ${custom_py} DESTINATION ${omg_install_path}/src)
      install(FILES ${custom_py} DESTINATION ${DDK_PATH}/ddk/ascendc/ops/impl/custom)
      file(GLOB custom_cpp ${CMAKE_CURRENT_SOURCE_DIR}/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/customize/op_impl/*/tbe/customize_impl/dynamic/*.cpp)
      install(FILES ${custom_cpp} DESTINATION ${omg_install_path}/src)
      install(FILES ${custom_cpp} DESTINATION ${DDK_PATH}/ddk/ascendc/ops/impl/custom)
      file(GLOB custom_h ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.h)
      install(FILES ${custom_h} DESTINATION ${omg_install_path}/src)
      install(FILES ${custom_h} DESTINATION ${DDK_PATH}/ddk/ascendc/ops/impl/custom)
    endforeach()

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/build_out/libcustom_op.so DESTINATION ${omg_install_path}/lib)

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/build_out/npu_custom_opinfo.json DESTINATION ${omg_install_path}/config)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/build_out/npu_custom_opinfo.json DESTINATION ${DDK_PATH}/tools/tools_omg/master/lib64/config)

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/customize/op_impl/ai_core/tbe/config/${compute_unit}/aic-${compute_unit}-ops-info.json DESTINATION ${omg_install_path}/info)
  endforeach()
endfunction()
