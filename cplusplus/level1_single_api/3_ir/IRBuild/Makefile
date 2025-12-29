ifeq ($(ASCEND_HOME_PATH),)
    ASCEND_PATH := /usr/local/Ascend/cann
else
    ASCEND_PATH := $(ASCEND_HOME_PATH)
endif

LOCAL_DIR  := ./

LOCAL_MODULE_NAME := ir_build
CC := g++
CFLAGS := -std=c++11 -g -Wall -D_GLIBCXX_USE_CXX11_ABI=0
SRCS := $(wildcard $(LOCAL_DIR)/src/main.cpp)

INCLUDES := -I $(ASCEND_PATH)/opp/built-in/op_proto/inc \
            -I $(ASCEND_PATH)/include \
            -I $(ASCEND_PATH)/include/ge \
            -I $(ASCEND_PATH)/include/graph \
            -I $(ASCEND_PATH)/include/parser \
            -I $(ASCEND_PATH)/include/external \

LIBS := -L $(ASCEND_PATH)/lib64 \
    -lgraph \
    -lgraph_base \
    -lfmk_parser \
    -lge_compiler \

ir_build:
	mkdir -p out
	$(CC) $(SRCS) $(INCLUDES) $(LIBS) $(CFLAGS) -o ./out/$(LOCAL_MODULE_NAME)

clean:
