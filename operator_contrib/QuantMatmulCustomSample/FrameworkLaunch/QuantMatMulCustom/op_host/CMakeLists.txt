
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)

opbuild(OPS_SRC ${ops_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)
