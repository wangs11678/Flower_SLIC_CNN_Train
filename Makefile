CXXFLAGS = $(shell pkg-config --cflags opencv)
LDLIBS = $(shell pkg-config --libs opencv)

LFLAGS := -pthread -lnsl -lX11  
LFLAGS += -lcaffe -lglog -lgflags \
    -lprotobuf -lboost_system -lboost_filesystem \
    -lboost_thread -lboost_coroutine -lboost_context -lboost_regex\
    -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect\
    -lstdc++ -lcblas -latlas  
  
LFLAGS += -Wl,-rpath=$(CAFFE_BUILD_LIB)  

CC = g++
CAFFEROOT = /home/wangs/caffe

objects = caffe_net_memorylayer.h extract_feature.h extract_feature.cpp utils.h utils.cpp main.cpp \
		  train.h train.cpp predict.h predict.cpp \
		  liblinear/linear.cpp liblinear/tron.cpp liblinear/blas/daxpy.c liblinear/blas/ddot.c \
          liblinear/blas/dnrm2.c liblinear/blas/dscal.c

classify : $(objects)
	$(CC) -o classify $(objects) $(CXXFLAGS) $(LDLIBS) $(LFLAGS) -I$(CAFFEROOT)/include/ -D CPU_ONLY \
	-I$(CAFFEROOT)/.build_release/src/ -L$(CAFFEROOT)/build/lib/ -lcaffe
	
clean:
	/bin/rm -f classify *.o

clean-all: clean
	/bin/rm -f *~ 

