cmake_minimum_required(VERSION 3.16.0)
project(opp)

set(DDK_PATH $ENV{DDK_PATH})

if (HOST)
    set(CMAKE_COMPILE ${CMAKE_CXX_COMPILER})

    set(ASCEND_AUTOGEN_PATH ${CMAKE_BINARY_DIR}/autogen)

    file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_PATH})
    set(CUSTOM_COMPILE_OPTIONS "custom_compile_options.ini")
    execute_process(COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
                    COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS})

    include(cmake/func.cmake)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
        add_subdirectory(op_host)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
        add_subdirectory(op_kernel)
    endif()

    get_system_info(SYSTEM_INFO)

    # CPack config
    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack opp project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack opp project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
    set(CPACK_PACKAGE_FILE_NAME "custom_opp_${SYSTEM_INFO}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/makeself.cmake)
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)

    #host
    build_host()
else()
    include(cmake/func.cmake)
    if(INSTALL)
        #install
        install_ddk()
    elseif(NOT "${ANDROID_NDK}" STREQUAL "" OR NOT "${OHOS_PLATFORM}" STREQUAL "")
        #ddk
        if (NOT "${ANDROID_NDK}" STREQUAL "")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_ABI_NAMESPACE=__1")
        endif()
        build_ddk()
    endif()
endif()